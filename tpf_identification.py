import warnings
from typing import Tuple, Union

from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from astropy.table import QTable
from astropy.time import Time
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import Wcsprm
from astroquery.gaia import Gaia
from astroquery.hips2fits import hips2fits
from erfa import ErfaWarning
from lightkurve import TessTargetPixelFile, KeplerTargetPixelFile
from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.colorbar import Colorbar
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

Gaia.MAIN_GAIA_TABLE = "gaiadr3.gaia_source"
Gaia.ROW_LIMIT = -1
REF_EPOCH = Time("J2016")


def calculate_theta(w: Wcsprm) -> Tuple[float, bool]:
    """
    Calculates the rotation angle of TPF according to the wcs in the FITS Header.

    Parameters
    ----------
    w: `~astropy.wcs.Wcsprm`

    Returns
    -------
    theta: float
        Rotation angle of TPF [degree]
    reverse: bool
        Whether the direction of the TPF is reversed
    """

    pc = w.pc
    cdelt = w.cdelt
    cd = cdelt * pc

    det = np.linalg.det(cd)
    sgn = np.sign(det)

    theta = -np.arctan2(sgn * cd[0, 1], sgn * cd[0, 0])
    cdelt1 = sgn * np.sqrt(cd[0, 0] ** 2 + cd[0, 1] ** 2)

    if cdelt1 < 0:
        return theta + np.pi, True
    else:
        return theta, False


def add_orientation(ax: Axes, theta: float, pad: float, color: str, reverse: bool):
    """
    Plot the orientation arrows.

    Parameters
    ----------
    ax: `matplotlib.axes.Axes`
        A matplotlib axes object to plot into
    theta: float
        Rotation angle of TPF [degree]
    pad: float
        The padding between the arrow base and the edge of the axes
    color: str
        The color of the orientation arrows
    reverse: bool
        Whether the direction of the TPF is reversed
    """

    def get_arrow_loc():
        return pad * np.cos(theta) * 0.45 * ratio, pad * np.sin(theta) * 0.45

    def get_text_loc(x, y):
        return 1 - (pad * ratio + 1.7 * x), pad + 1.7 * y

    ratio = ax.get_data_ratio()
    x1, y1 = get_arrow_loc()
    theta += -np.pi / 2 if reverse else np.pi / 2
    x2, y2 = get_arrow_loc()

    ax.arrow((1 - pad * ratio), pad, -x1, y1, color=color, head_width=0.01, transform=ax.transAxes, zorder=100)
    ax.text(*get_text_loc(x1, y1), s="E", color=color, ha="center", va="center", transform=ax.transAxes, zorder=100)

    ax.arrow((1 - pad * ratio), pad, -x2, y2, color=color, head_width=0.01, transform=ax.transAxes, zorder=100)
    ax.text(*get_text_loc(x2, y2), s="N", color=color, ha="center", va="center", transform=ax.transAxes, zorder=100)


def add_scalebar(ax: Axes, pad: float, length: float, scale: str):
    """
    Plot the scale bar.

    Parameters
    ----------
    ax: `matplotlib.axes.Axes`
        A matplotlib axes object to plot into
    pad: float
        The padding between the left endpoint of the scale bar and the edge of the axes
    length: float
        The length of the scale bar
    scale: str
        Scale of the scale bar
    """

    ratio = ax.get_data_ratio()
    x_min = pad * ratio / 2
    x_max = x_min + length
    y_min = pad * 0.95
    y_max = pad * 1.05
    ax.hlines(y=pad, xmin=x_min, xmax=x_max, colors="k", ls="-", lw=1.5, transform=ax.transAxes)
    ax.vlines(x=x_min, ymin=y_min, ymax=y_max, colors="k", ls="-", lw=1.5, transform=ax.transAxes)
    ax.vlines(x=x_max, ymin=y_min, ymax=y_max, colors="k", ls="-", lw=1.5, transform=ax.transAxes)
    ax.text(x_min + length / 2, pad * 1.1, scale, horizontalalignment="center", fontsize=10, transform=ax.transAxes)


def query_nearby_gaia_objects(
    tpf: Union[TessTargetPixelFile, KeplerTargetPixelFile], tpf_radius: float, magnitude_limit: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Query the objects in the area of TPF from Gaia Catalog (Now Gaia DR3).

    Parameters
    ----------
    tpf: `lightkurve.TessTargetPixelFile` or `lightkurve.KeplerTargetPixelFile`
        Target pixel files read by `lightkurve`
    tpf_radius: float
        The radius of the TPF [arcsec]
    magnitude_limit: int or float
        The maximum magnitude limit of the stars

    Returns
    -------
    x: 1-D ndarray
        The x coordinates of the stars in pixel
    y: 1-D ndarray
        The y coordinates of the stars in pixel
    mag: 1-D ndarray
        The mean magnitudes of the stars in Gaia G band
    target_index: int
        The index of the target in the returned lists
    """

    # Get the parameters of the Gaia sources
    coord = SkyCoord(tpf.ra, tpf.dec, unit="deg", frame="icrs", equinox="J2000")
    radius = u.Quantity(tpf_radius * 1.5, u.arcsec)
    j = Gaia.cone_search_async(coord, radius)
    r = j.get_results()

    target_gaia_id = r[r["dist"] < 5 / 3600]["source_id"][0]
    print("Found in Gaia DR3. Source ID: {}".format(target_gaia_id))

    r.sort("phot_g_mean_mag")
    target_index = np.nonzero(r["source_id"] == target_gaia_id)[0][0]
    magnitude_limit = max(r["phot_g_mean_mag"][0] + 3, magnitude_limit)
    r = r[r["phot_g_mean_mag"] < magnitude_limit][: max(target_index + 50, 300)]

    qr = QTable([r["ra"].filled(), r["dec"].filled(), r["pmra"].filled(0), r["pmdec"].filled(0)])
    coords_gaia = SkyCoord(
        qr["ra"], qr["dec"], pm_ra_cosdec=qr["pmra"], pm_dec=qr["pmdec"], frame="icrs", obstime=REF_EPOCH
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ErfaWarning)
        coords_tess = coords_gaia.apply_space_motion(new_obstime=tpf.time[0])

    x, y = tpf.wcs.world_to_pixel(coords_tess)
    mag = np.asarray(r["phot_g_mean_mag"])

    return x, y, mag, target_index


def query_sky_img(
    tpf: Union[TessTargetPixelFile, KeplerTargetPixelFile],
    theta: float,
    x_length: float,
    y_length: float,
    reverse: bool,
) -> np.ndarray:
    """
    Query the image of the area of TPF from DSS2 Red Survey.

    Parameters
    ----------
    tpf: `lightkurve.TessTargetPixelFile` or `lightkurve.KeplerTargetPixelFile`
        Target pixel files read by `lightkurve`
    theta: float
        Rotation angle of TPF [degree]
    x_length: float
        The x length of the TPF [arcsecond]
    y_length: float
        The y length of the TPF [arcsecond]
    reverse: bool
        Whether the direction of the TPF is reversed

    Returns
    -------
    2-D array
    """

    def query_sky_data(hips):
        return hips2fits.query(
            hips=hips,
            width=n_pixel,
            height=n_pixel,
            projection="TAN",
            ra=center_ra * u.deg,
            dec=center_dec * u.deg,
            fov=radius * u.arcsec,
            format="png",
            rotation_angle=Angle(-theta * u.rad),
            cmap="Greys",
        )

    n_pixel = 200 if tpf.meta["TELESCOP"] == "TESS" else 50

    center_ra, center_dec = tpf.wcs.all_pix2world([(tpf.shape[1:][1] + 1) / 2], [(tpf.shape[1:][0] + 1) / 2], 1)

    if reverse:
        theta = np.pi - theta

    radius = 1.5 * max(x_length, y_length)

    try:
        sky_data = query_sky_data("CDS/P/DSS2/red")
    except Exception:
        sky_data = query_sky_data("CDS/P/DSS2/NIR")

    if reverse:
        sky_data = np.flip(sky_data, axis=0)
    else:
        sky_data = np.flip(sky_data, axis=(0, 1))

    x_pixel_sky = n_pixel * x_length / radius
    x_start = int((n_pixel - x_pixel_sky) / 2)
    x_stop = n_pixel - x_start

    y_pixel_sky = n_pixel * y_length / radius
    y_start = int((n_pixel - y_pixel_sky) / 2)
    y_stop = n_pixel - y_start

    return sky_data[y_start:y_stop, x_start:x_stop]


def plot_identification(ax: Axes, tpf: Union[TessTargetPixelFile, KeplerTargetPixelFile]):
    """
    Plot the identification charts.

    Parameters
    ----------
    ax: `matplotlib.axes.Axes`
        A matplotlib axes object to plot into
    tpf: `lightkurve.TessTargetPixelFile` or `lightkurve.KeplerTargetPixelFile`
        Target pixel files read by `lightkurve`
    """

    divider = make_axes_locatable(ax)
    ax_tpf = divider.append_axes("right", size="100%", pad=0.1)
    ax_cb = divider.append_axes("right", size="8%", pad=0.35)
    ax_sky = ax

    # Use pixel scale for query size
    pixel_scale = 21 if tpf.meta["TELESCOP"] == "TESS" else 4

    x_pixel, y_pixel = tpf.shape[1:][1], tpf.shape[1:][0]
    x_length = x_pixel * pixel_scale
    y_length = y_pixel * pixel_scale
    tpf_radius = max(x_length, y_length)
    theta, reverse = calculate_theta(tpf.wcs.wcs)

    # TPF plot
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median_flux = np.nanmedian(tpf.flux.value, axis=0)

    try:
        division = int(np.log10(np.nanmax(median_flux)))
    except ValueError:
        division = 0

    image = median_flux / 10**division

    splot = ax_tpf.imshow(image, norm=ImageNormalize(stretch=LogStretch()), origin="lower", cmap="viridis", zorder=0)
    ax_tpf.set_xlim([-0.5, x_pixel - 0.5])
    ax_tpf.set_ylim([-0.5, y_pixel - 0.5])
    ax_tpf.set_xticks(np.arange(0, x_pixel, 1))
    ax_tpf.set_yticks(np.arange(0, y_pixel, 1))
    ax_tpf.set_xticklabels(np.arange(1, x_pixel + 1, 1))
    ax_tpf.set_yticklabels(np.arange(1, y_pixel + 1, 1))
    ax_tpf.yaxis.set_ticks_position("right")
    ax_tpf.invert_xaxis()

    try:
        x, y, gaia_mags, this = query_nearby_gaia_objects(tpf, tpf_radius, 18)
        target_gaia_mag = gaia_mags[this]

        size_k = 1.2 * np.piecewise(
            target_gaia_mag,
            [target_gaia_mag < 12, 12 <= target_gaia_mag < 18, target_gaia_mag > 18],
            [70, lambda mag: 190 - mag * 10, 10],
        )
        if tpf.meta["TELESCOP"] != "TESS":
            size_k = size_k * 5
        size = size_k / 1.5 ** (gaia_mags - target_gaia_mag)

        ax_tpf.scatter(x, y, s=size, c="red", alpha=0.5, edgecolor=None, zorder=11)
        ax_tpf.scatter(x[this], y[this], marker="x", c="white", s=size_k / 2.5, zorder=12)

    except IndexError:
        print("Not found in Gaia DR3!")
        at = AnchoredText("No Gaia DR3 Data", frameon=False, loc="upper left", prop=dict(size=13))
        ax_tpf.add_artist(at)

    # Pipeline aperture
    aperture = tpf.pipeline_mask
    for i in range(aperture.shape[0]):
        for j in range(aperture.shape[1]):
            xy = (j - 0.5, i - 0.5)
            if aperture[i, j]:
                ax_tpf.add_patch(patches.Rectangle(xy, 1, 1, color="tomato", fill=True, alpha=0.4))
                ax_tpf.add_patch(patches.Rectangle(xy, 1, 1, color="tomato", fill=False, alpha=0.6, lw=1.5, zorder=9))
            else:
                ax_tpf.add_patch(patches.Rectangle(xy, 1, 1, color="gray", fill=False, alpha=0.2, lw=0.5, zorder=8))

    # Sky
    sky_img = query_sky_img(tpf, theta, x_length, y_length, reverse)

    ax_sky.imshow(sky_img, origin="lower")
    ax_sky.set_xticks([])
    ax_sky.set_yticks([])
    ax_sky.set_xticklabels([])
    ax_sky.set_yticklabels([])
    ax_sky.set_xlim(0, sky_img.shape[1])
    ax_sky.set_ylim(0, sky_img.shape[0])
    ax_sky.invert_xaxis()

    at = AnchoredText(tpf.meta["OBJECT"], frameon=False, loc="upper left", prop=dict(size=13))
    ax_sky.add_artist(at)

    # Add orientation arrows
    add_orientation(ax=ax_tpf, theta=theta, pad=0.15, color="w", reverse=reverse)
    add_orientation(ax=ax_sky, theta=theta, pad=0.15, color="k", reverse=reverse)

    # Add scale bar
    add_scalebar(ax=ax_sky, pad=0.15, length=1 / x_pixel, scale='{}"'.format(pixel_scale))

    # Add color bar
    cb = Colorbar(ax=ax_cb, mappable=splot, orientation="vertical", ticklocation="right")

    max_diff = int((np.nanmax(image) - np.nanmin(image)) / 0.01)
    n_ticks = min(4, max_diff)
    cbar_ticks = np.linspace(np.nanmin(image), np.nanmax(image), n_ticks, endpoint=True)

    cb.set_ticks(cbar_ticks)
    cb.set_ticklabels(np.round(cbar_ticks, 2))
    exponent = r"$\times$ $10^{}$".format(division)
    cb.set_label(r"Flux {} (e$^-$)".format(exponent), fontsize=13)


if __name__ == "__main__":
    from pathlib import Path
    import lightkurve as lk
    from matplotlib import pyplot as plt

    result_path = Path.cwd() / "results"
    result_path.mkdir(exist_ok=True)

    file_path_list = sorted([i for i in (Path.cwd() / "tpfs").glob("*.fit*")])

    index = 0
    for file_path in file_path_list:
        print("-------------------------")
        tpf = lk.read(file_path)

        index += 1
        print("Num {}, {}".format(index, tpf.meta["OBJECT"]))

        mission = tpf.mission
        if mission == "TESS":
            part = "S{:0>2d}".format(tpf.meta["SECTOR"])
        elif mission == "K2":
            part = "C{:0>2d}".format(tpf.meta["CAMPAIGN"])
        elif mission == "Kepler":
            part = "Q{:0>2d}".format(tpf.meta["QUARTER"])
        else:
            raise ValueError

        fig, ax = plt.subplots(figsize=(9, 4))
        plot_identification(ax, tpf)

        plt.savefig(
            result_path / Path("{}-{}.pdf".format(tpf.meta["OBJECT"].replace(" ", ""), part)), bbox_inches="tight"
        )
