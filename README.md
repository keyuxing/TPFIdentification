# TPFIdentification
**Plot identification charts for Kepler, K2 and TESS.**

![alt text](https://github.com/keyuxing/tpf_identification/blob/main/examples/TIC150428135-S01.jpg)

![alt text](https://github.com/keyuxing/tpf_identification/blob/main/examples/KIC1161345-Q17.jpg)

This identification chart is useful to check compact candidates whether they are in the 
location containing numerous stars since the nearby stars will contaminate the light to 
the target. In each module, the right panel presents the target, marked by cross symbol,
taken from TESS target pixel files. The size of the circle indicates the relative 
brightness of the stars. The left panel presents the same sky coverage but from 
[DSS2 Red 2 survey](https://skyview.gsfc.nasa.gov/current/cgi/moreinfo.pl?survey=DSS2%20Red).

This repository is revised based on 
[_tpfplotter_](https://github.com/jlillo/tpfplotter). 

## Requirements
Running this script requires
[_astropy_](https://github.com/astropy/astropy), 
[_astroquery_](https://github.com/astropy/astroquery),
[_matplotlib_](https://github.com/matplotlib/matplotlib), 
and [_lightkurve_](https://github.com/lightkurve/lightkurve)
to be installed.

All you need is to install [_lightkurve_](https://github.com/lightkurve/lightkurve)
using `pip`:
```shell
pip install lightkurve
```
or `conda`:
```shell
conda install lightkurve
```
And other required packages will be installed with it.  

## How to use
Clone or download this repository to your computer, and then download the TPFs
you want to plot identification charts to the `tpf_identification/tpfs` folder.

Run the command below in the `tpf_identification` folder to plot identification charts.
```
python tpf_identification.py
```
