from astropy import units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import vstack
from astropy.visualization import LogStretch
from astropy.visualization.mpl_normalize import ImageNormalize
from astropy.wcs import Wcsprm
from astroquery.simbad import Simbad
from astroquery.vizier import Vizier
from astroquery.mast import Catalogs
from astroquery.hips2fits import hips2fits
from lightkurve.targetpixelfile import KeplerTargetPixelFile, TessTargetPixelFile
from matplotlib import patches
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

Vizier.ROW_LIMIT = -1


def calculate_theta(w: Wcsprm):
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
        reverse = True
        theta += np.pi
    else:
        reverse = False

    return theta, reverse


def add_orientation(theta, ax, head_width, color, x0, reverse, shift=False):
    y0 = x0
    x1, y1 = x0 * np.cos(theta) * 0.5, y0 * np.sin(theta) * 0.5

    if not reverse:
        theta += np.pi / 2
    else:
        theta -= np.pi / 2

    x2, y2 = x0 * np.cos(theta) * 0.5, y0 * np.sin(theta) * 0.5
    if shift:
        x0 -= 0.5
        y0 -= 0.5

    ax.arrow(x0, y0, x1, y1, head_width=head_width, color=color, zorder=100)
    ax.text(x0 + 1.6 * x1, y0 + 1.6 * y1, "E", color=color, ha="center", va="center", zorder=100)

    ax.arrow(x0, y0, x2, y2, head_width=head_width, color=color, zorder=100)
    ax.text(x0 + 1.6 * x2, y0 + 1.6 * y2, "N", color=color, ha="center", va="center", zorder=100)


def add_scalebar(ax, length, text, x0, y0):
    ax.hlines(y=y0, xmin=x0, xmax=x0 + length, colors="black", ls="-", lw=1.5, label="%d km" % length)
    ax.vlines(x=x0, ymin=y0 * 0.9, ymax=y0 * 1.1, colors="black", ls="-", lw=1.5)
    ax.vlines(x=x0 + length, ymin=y0 * 0.9, ymax=y0 * 1.1, colors="black", ls="-", lw=1.5)
    ax.text(x0 + length / 2, y0 * 1.2, text + '"', horizontalalignment="center", fontsize=10)


def add_gaia_figure_elements(tpf, gaia_id, magnitude_limit, tpf_radius, search_scale):
    # Get the positions of the Gaia sources
    c1 = SkyCoord(tpf.ra * u.deg, tpf.dec * u.deg, frame="icrs")
    search_radius = Angle(search_scale * tpf_radius * u.arcsec)

    result = Vizier.query_region(c1, catalog=["I/345/gaia2"], radius=search_radius)["I/345/gaia2"]
    try:
        if not (result["Source"] == int(gaia_id)).any():
            result_k = Vizier.query_object("Gaia DR2 " + gaia_id, catalog=["I/345/gaia2"])["I/345/gaia2"]
            result_k = result_k[result_k["Source"] == int(gaia_id)]
            result = vstack([result, result_k])

        result = result.to_pandas()
        result = result.sort_values(by="Gmag", ignore_index=True)
        this = result[result.Source == int(gaia_id)].index[0]
        result = result[result.Gmag < magnitude_limit].iloc[: max(this + 50, 300)]

        year = ((tpf.time[0].jd - 2457206.375) * u.day).to(u.year)
        pm_ra = ((np.nan_to_num(np.asarray(result.pmRA)) * u.milliarcsecond / u.year) * year).to(u.degree).value
        pm_dec = ((np.nan_to_num(np.asarray(result.pmDE)) * u.milliarcsecond / u.year) * year).to(u.degree).value
        result.RA_ICRS += pm_ra
        result.DE_ICRS += pm_dec
        coords = tpf.wcs.all_world2pix(np.vstack([result.RA_ICRS, result.DE_ICRS]).T, 0)
        return coords, result["Gmag"], this

    except TypeError:
        return None, None, None


def get_gaia_data(tpf):
    """
    Get Gaia parameters

    Returns
    -----------------------
    GaiaID, Gaia_mag
    """
    try:
        # Query gaia directly from target label
        label = tpf.get_header(ext=0).get("OBJECT")
        source_ids = Simbad.query_objectids(label)["ID"]
        for i in range(len(source_ids)):
            if "Gaia DR2" in source_ids[i]:
                gaia_id = source_ids[i].split(" ")[-1]
                r = Catalogs.query_object("Gaia DR2 " + gaia_id, catalog="Gaia", data_release="DR2", radius=0.005)
                gaia_mag = r[r["source_id"] == gaia_id]["phot_g_mean_mag"][0]
                return gaia_id, gaia_mag
    except TypeError:
        pass

    # Query gaia from target coordinate
    print("Get gaia data from target id failed, using coordinate instead...")
    ra = tpf.get_header(ext=0).get("RA_OBJ")
    dec = tpf.get_header(ext=0).get("DEC_OBJ")
    target_coord = SkyCoord(ra, dec, frame="icrs", unit="deg")

    query_success = False
    query_region = 10
    attempts = 0
    while not query_success and attempts <= 2:
        result = Vizier.query_region(target_coord, catalog=["I/345/gaia2"], radius=Angle(query_region, "arcsec"))
        try:
            result = result["I/345/gaia2"]
            query_success = True
            dist = np.sqrt((result["RA_ICRS"] - ra) ** 2 + (result["DE_ICRS"] - dec) ** 2)
            idx = np.argmin(dist)
            return result[idx]["Source"], result[idx]["Gmag"]
        except Exception:
            query_region += 5
            attempts += 1
    return None, None


def get_sky_img(tpf, theta, x_length, y_length, tpf_radius, reverse):
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

    radius = 1.5 * tpf_radius

    if tpf.mission == "Kepler" or tpf.mission == "K2":
        n_pixel = 50
    elif tpf.mission == "TESS":
        n_pixel = 200
    else:
        raise ValueError

    center_ra, center_dec = tpf.wcs.all_pix2world([(tpf.shape[1:][1] + 1) / 2], [(tpf.shape[1:][0] + 1) / 2], 1)

    if reverse:
        theta = np.pi - theta

    try:
        sky_data = query_sky_data("CDS/P/DSS2/red")
    except Exception:
        sky_data = query_sky_data("CDS/P/DSS2/NIR")

    if reverse:
        sky_data = np.flip(sky_data, axis=0)
    else:
        sky_data = np.flip(sky_data, axis=(0, 1))

    x_pixel_rotated = n_pixel * y_length / radius
    x_start = int((n_pixel - x_pixel_rotated) / 2)
    x_stop = n_pixel - x_start

    y_pixel_rotated = n_pixel * x_length / radius
    y_start = int((n_pixel - y_pixel_rotated) / 2)
    y_stop = n_pixel - y_start

    sky_img = sky_data[x_start:x_stop, y_start:y_stop]

    return x_pixel_rotated, y_pixel_rotated, sky_img


def plot_identification(ax_sky, tpf: KeplerTargetPixelFile | TessTargetPixelFile):
    divider = make_axes_locatable(ax_sky)
    ax_tpf = divider.append_axes("right", size="100%", pad=0.1)
    ax_cb = divider.append_axes("right", size="8%", pad=0.35)

    # Use pixel scale for query size
    telescope = tpf.get_header(ext=0).get("TELESCOP").strip()
    if telescope == "TESS":
        pixel_scale = 21.0
    elif telescope == "Kepler":
        pixel_scale = 4.0
    else:
        raise ValueError

    x_pixel, y_pixel = tpf.shape[1:][1], tpf.shape[1:][0]
    x_length = x_pixel * pixel_scale
    y_length = y_pixel * pixel_scale
    tpf_radius = max(tpf.shape[1:]) * pixel_scale
    theta, reverse = calculate_theta(tpf.wcs.wcs)

    # TPF plot
    norm = ImageNormalize(stretch=LogStretch())
    median_flux = np.nanmedian(tpf.flux.value, axis=0)

    try:
        division = int(np.log10(np.nanmax(median_flux)))
    except ValueError:
        division = 0

    image = median_flux / 10**division

    splot = ax_tpf.imshow(image, norm=norm, origin="lower", cmap="viridis", zorder=0)
    ax_tpf.set_xlim([-0.5, x_pixel - 0.5])
    ax_tpf.set_ylim([-0.5, y_pixel - 0.5])
    ax_tpf.set_xticks(np.arange(0, x_pixel, 1))
    ax_tpf.set_yticks(np.arange(0, y_pixel, 1))
    ax_tpf.set_xticklabels(np.arange(1, x_pixel + 1, 1))
    ax_tpf.set_yticklabels(np.arange(1, y_pixel + 1, 1))
    ax_tpf.yaxis.set_ticks_position("right")
    ax_tpf.invert_xaxis()

    # Get Gaia ID and mag of target
    target_gaia_id, target_gaia_mag = get_gaia_data(tpf)

    add_gaia = False
    if target_gaia_id is not None:
        print("Found in Gaia DR2, Gaia ID: {}".format(target_gaia_id))
        add_elements_attempt = 0
        search_scale = 1.0
        this = None
        while add_elements_attempt <= 10 and this is None:
            # Make the Gaia Figure Elements
            try:
                coords, gaia_mags, this = add_gaia_figure_elements(
                    tpf, target_gaia_id, max(target_gaia_mag + 2.0, 18), tpf_radius, search_scale
                )
                x, y = coords[:, 0], coords[:, 1]
                size_k = 1.2 * np.piecewise(
                    target_gaia_mag,
                    [target_gaia_mag < 12, 12 <= target_gaia_mag < 18, target_gaia_mag > 18],
                    [70, lambda mag: 190 - mag * 10, 10],
                )
                if telescope == "Kepler":
                    size_k = size_k * 5
                size = size_k / 1.5 ** (gaia_mags - target_gaia_mag)

                ax_tpf.scatter(x, y, s=size, c="red", alpha=0.5, edgecolor=None, zorder=11)
                ax_tpf.scatter(x[this], y[this], marker="x", c="white", s=size_k / 2, zorder=12)
                add_gaia = True

            except TypeError:
                if not add_elements_attempt:
                    print("Add gaia figure elements failed, retrying...")
                search_scale += 0.05
                add_elements_attempt += 1

    if not add_gaia:
        print("Not found in Gaia DR2")
        ax_tpf.text(0.95 * x_pixel, 0.85 * y_pixel, "No Gaia DR2 Data", fontsize=14)

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
    y_pixel_sky, x_pixel_sky, sky_img = get_sky_img(tpf, theta, x_length, y_length, tpf_radius, reverse)

    ax_sky.imshow(sky_img, origin="lower")
    ax_sky.set_xticks([])
    ax_sky.set_yticks([])
    ax_sky.set_xticklabels([])
    ax_sky.set_yticklabels([])
    ax_sky.set_xlim([0, x_pixel_sky])
    ax_sky.set_ylim([0, y_pixel_sky])
    ax_sky.invert_xaxis()
    ax_sky.text(0.96 * x_pixel_sky, 0.92 * y_pixel_sky, tpf.get_header(ext=0).get("OBJECT"), fontsize=13)

    # Add orientation and scalebar
    add_orientation(theta, ax_tpf, x_pixel / 100, "white", 0.15 * x_pixel, reverse, shift=True)
    add_orientation(theta, ax_sky, x_pixel_sky / 100, "black", 0.15 * x_pixel_sky, reverse)

    if telescope == "TESS":
        add_scalebar(ax_sky, 20 / tpf_radius * x_pixel_sky, "20", 0.8 * x_pixel_sky, 0.1 * y_pixel_sky)
    elif telescope == "Kepler":
        add_scalebar(ax_sky, 4 / tpf_radius * x_pixel_sky, "4", 0.8 * x_pixel_sky, 0.1 * y_pixel_sky)

    # Colorbar
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
    from astropy.units import UnitsWarning
    import lightkurve as lk
    from matplotlib import pyplot as plt

    import warnings

    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UnitsWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    result_path = Path.cwd() / "results"
    result_path.mkdir(exist_ok=True)

    file_path_list = [i for i in (Path.cwd() / "tpfs").glob("*.fit*")]
    file_path_list.sort()

    index = 0
    for file_path in file_path_list:
        print("-------------------------")
        print(file_path)
        tpf = lk.read(file_path)

        index += 1
        label = tpf.get_header(ext=0).get("OBJECT")
        print("Num {}, {}".format(index, label))

        mission = tpf.mission
        if mission == "TESS":
            part = "S{:0>2d}".format(tpf.get_header(ext=0).get("SECTOR"))
        elif mission == "K2":
            part = "C{:0>2d}".format(tpf.get_header(ext=0).get("CAMPAIGN"))
        elif mission == "Kepler":
            part = "Q{:0>2d}".format(tpf.get_header(ext=0).get("QUARTER"))
        else:
            raise ValueError

        fig, ax = plt.subplots(figsize=(9, 4))
        plot_identification(ax, tpf)

        plt.savefig(result_path / Path("{}-{}.pdf".format(label.replace(" ", ""), part)), bbox_inches="tight")
