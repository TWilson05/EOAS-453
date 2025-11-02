# -----------------------------------------------------------------------------
# Super-volcano Snowball Model (ChatGPT drafted)
# -----------------------------------------------------------------------------
# 1) Build 30°×30° lon/lat grids (12×6 cells).
# 2) Initializes land/ocean/ice fractions either from Natural Earth (NE1) or
#    from Tyler's values.
# 3) Builds a simple energy-balance model (EBM) with:
#    - Shortwave in (with volcanic attenuation φ),
#    - Longwave out (σT^4 with the scaled form),
#    - Heat exchange with lateral grids,
#    - Changing sea-ice fraction that affects albedo & heat capacity.
# 4) Outputs two synchronized interactive maps (temp + ocean fraction) over a
#    basemap.
# -----------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import os

# -----------------------------------------------------------------------------
# A robust version of reading basemap .tif. Rasterio is tried first.
# No-clue bugs encountered when importing the map, suggested by ChatGPT.
# -----------------------------------------------------------------------------
try:
    import rasterio
except Exception:
    rasterio = None
try:
    from PIL import Image
except Exception:
    Image = None

# -----------------------------------------------------------------------------
# Simulation horizon and I/O
# -----------------------------------------------------------------------------
YEARS = 33.0                      # Total length of the time series / Default: 33 years / Float, yerra
DT_DAYS = 1.0                     # Time step / Default: 1 days / Float, days
NE1_PATHS = ["NE1_50M_SR_W.tif"]  # Search order for paths, can add more to the end
USE_BASEMAP = True                # Turn on basemap or not (False for quicker debugging)

# -----------------------------------------------------------------------------
# Volcanic forcing φ(t, lon, lat)
# -----------------------------------------------------------------------------
# Scenario: A big eruption that:
#  - starts at t0_days, two stage: rise and decay,
#  - spreads to (sigma_latmax, sigma_lonmax) over tau_spread_days,
#  - optional hemi_bias (more aerosol in the hemisphere).
# Default values are for Mt Baker, which makes sense as it is a stratovolcano near subduction zone
ERUPTIONS = [dict(t0_days=500.0, lat=48.7767, lon=-121.8144, amplitude=0.5,  # Default: 500, 48.7767, -121.8144, 0.5
                  tau_rise_days=60.0, tau_decay_days=730.0,          # 60, 730
                  sigma_lat0_deg=15.0, sigma_latmax_deg=60.0,        # 15, 60
                  sigma_lon0_deg=30.0, sigma_lonmax_deg=180.0,       # 30, 180
                  tau_spread_days=150.0, hemi_bias=0.4)]             # 150, 0.4

# -----------------------------------------------------------------------------
# Physical constants
# -----------------------------------------------------------------------------
# Stefan–Boltzmann constant in a numerically stable scaled form to avoid overflow for large T^4
sigma_B_scaled = 5.6696
solar_const = 1368.0
r_earth = 6371e3
area_earth = np.pi * r_earth**2

# Simple one-layer atmosphere terms
transmissivity_atm = 0.63
albedo_sky = 0.20

# Albedos for the three surface types
albedo_surface = 0.40
albedo_ocean   = 0.10
albedo_ice     = 0.60

# Properties for each surface type
density_land, density_water, density_ice = 2500.0, 1028.0, 900.0
depth_land,   depth_ocean,   depth_ice   = 1.0,    70.0,    1.0
spec_heat_land, spec_heat_water, spec_heat_ice = 790.0, 4187.0, 2060.0

# -----------------------------------------------------------------------------
# Grid definition (12 × 6, 30° spacing)
# -----------------------------------------------------------------------------
lat_edges = np.arange(-90, 91, 30)
lon_edges = np.arange(-180, 181, 30)
nlon = len(lon_edges)-1; nlat = len(lat_edges)-1
lat_centers = 0.5*(lat_edges[:-1]+lat_edges[1:])
lon_centers = 0.5*(lon_edges[:-1]+lon_edges[1:])

# Suggested by Tyler
area_fractions = np.array(12*[[0.0670, 0.1830, 0.2500, 0.2500, 0.1830, 0.0670]]) / 12
geometric_factors = np.array(12*[[0.1076, 0.2277, 0.3045, 0.3045, 0.2277, 0.1076]])

lengths_lat = np.array(12*[[2.0015e7, 3.4667e7, 4.0030e7, 3.4667e7, 2.0015e7]]) / 12
lengths_lon = np.array(12*[6*[np.pi * r_earth]]) / 6

k_vals_lat = np.ones((12,5)) * 1e7
k_vals_lon = np.ones((12,6)) * 1e7

# -----------------------------------------------------------------------------
# Find basemap path, downsample (Suggested by ChatGPT)
# -----------------------------------------------------------------------------
def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None

def _downsample_rgb(rgb, max_w=2048):
    """Shrink very large rasters so each frame update stays snappy."""
    H, W = rgb.shape[:2]
    if W <= max_w or Image is None:
        return rgb
    im = Image.fromarray(rgb)
    new_w = max_w
    new_h = int(H * (new_w/W))
    return np.array(im.resize((new_w, new_h), Image.BILINEAR))

# -----------------------------------------------------------------------------
# Land/Ocean/Ice fractions from NE1
# -----------------------------------------------------------------------------
def derive_fractions_from_ne1():
    """
    Returns:
      basemap_rgb  : RGB image to draw once behind the rasters (or None),
      land_fractions, ocean_fractions, ice_fractions : (12,6) arrays.
    Logic:
      * If NE1 is available, read grayscale brightness and use a simple
        threshold (gray > 150) as land (Tyler’s approach), then aggregate
        to 30°×30° bins. Add fixed polar ice fractions to the first/last
        latitude bands. Otherwise, use Tyler’s band-averaged fractions.
    """
    antarctic_ice = 0.7019
    arctic_ice    = 0.3670
    land_fractions = np.zeros((12,6))
    ocean_fractions = np.zeros((12,6))
    ice_fractions   = np.zeros((12,6))

    path = _first_existing(NE1_PATHS)
    basemap_rgb = None

    if USE_BASEMAP and path is not None:
        gray = None
        try:
            # Prefer rasterio; fall back to PIL
            if rasterio is not None:
                with rasterio.open(path) as src:
                    band1 = src.read(1)
                    gray = band1.astype(np.uint8)
                    if src.count >= 3:
                        rgb = np.moveaxis(src.read([1,2,3]), 0, -1).astype(np.uint8)
                        basemap_rgb = _downsample_rgb(rgb)
            if gray is None and Image is not None:
                im = Image.open(path).convert('L')
                gray = np.array(im, dtype=np.uint8)
                basemap_rgb = _downsample_rgb(np.array(im.convert('RGB')))
        except Exception:
            gray = None

        if gray is not None:
            # Build land mask from brightness and bin to the 30° grid
            H, W = gray.shape
            landmask = (gray > 150).astype(np.uint8)
            lons = np.linspace(-180, 180, W, endpoint=False)
            lats = np.linspace(  90, -90, H, endpoint=False)
            lon_bins = lon_edges; lat_bins = lat_edges
            frac_land = np.zeros((len(lat_bins)-1, len(lon_bins)-1))
            for i in range(len(lat_bins)-1):
                lat_mask = (lats <= lat_bins[i+1]) & (lats > lat_bins[i])
                for j in range(len(lon_bins)-1):
                    lon_mask = (lons >= lon_bins[j]) & (lons < lon_bins[j+1])
                    window = landmask[np.ix_(lat_mask, lon_mask)]
                    frac_land[i, j] = window.mean() if window.size else 0.0

            # Convert to our (12,6) orientation
            frac_water = 1.0 - frac_land
            land_fractions  = frac_land.T
            ocean_fractions = frac_water.T

            # Impose fixed polar sea-ice fractions (first & last latitude bands)
            land_fractions[:,0]  *= (1 - antarctic_ice)
            land_fractions[:,-1] *= (1 - arctic_ice)
            ocean_fractions[:,0]  *= (1 - antarctic_ice)
            ocean_fractions[:,-1] *= (1 - arctic_ice)
            ice_fractions[:,0] = antarctic_ice
            ice_fractions[:,-1] = arctic_ice
            return basemap_rgb, land_fractions, ocean_fractions, ice_fractions

    # Fallback: band means (Tyler)
    land_bands  = np.array([0.008, 0.0116, 0.2522, 0.3550, 0.5786, 0.0020])
    ocean_bands = np.array([0.2901, 0.9880, 0.7472, 0.6442, 0.4176, 0.6310])
    ice_bands   = np.array([0.7019, 0.0004, 0.0006, 0.0008, 0.0038, 0.3670])
    land_fractions  = np.tile(land_bands,  (12,1))
    ocean_fractions = np.tile(ocean_bands, (12,1))
    ice_fractions   = np.tile(ice_bands,   (12,1))
    return basemap_rgb, land_fractions, ocean_fractions, ice_fractions

# Fractions + flat staging array (3×(12*6)) used throughout the integrator
basemap_rgb, land_fractions, ocean_fractions, ice_fractions = derive_fractions_from_ne1()
frac_arr_init = np.vstack([land_fractions.flatten(),
                           ocean_fractions.flatten(),
                           ice_fractions.flatten()])

# Convenience helpers (left as identity wrappers to match Tyler’s pattern)
def compute_pcZ(p, c, z): return p*c*z
def compute_albedo(a): return a

# Parameter vectors for the three surface types
density_arr   = np.array([2500.0, 1028.0, 900.0])
spec_heat_arr = np.array([ 790.0, 4187.0, 2060.0])
depth_arr     = np.array([   1.0,   70.0,    1.0])
albedo_arr    = np.array([  0.40,   0.10,   0.60])

# Precompute per-cell bulk heat capacity (pcZ) and initial albedo
pcZ_vals = np.array([
    np.sum(frac_arr_init[:,i] * compute_pcZ(density_arr, spec_heat_arr, depth_arr))
    for i in range(frac_arr_init.shape[1])
]).reshape((12,6))
avg_albedos_init = np.array([
    np.sum(frac_arr_init[:,i] * compute_albedo(albedo_arr))
    for i in range(frac_arr_init.shape[1])
]).reshape((12,6))

# -----------------------------------------------------------------------------
# Sea-ice evolution (very simple “relaxation” style update)
# -----------------------------------------------------------------------------
def change_ice_frac(temp, frac_arr, change_per_deg=0.05/365.0):
    """
    Update ocean vs. ice partition based on temperature relative to 0°C:
      * If T < 273.15 K and there's open ocean, shift some ocean → ice.
      * If T > 273.15 K and there's ice, shift some ice → ocean.
    The per-degree rate (per day) is tunable via change_per_deg.
    Returns:
      new_frac (3×N), new_albedo (12×6) for the next step.
    """
    land_fr  = frac_arr[0].reshape(temp.shape).copy()
    ocean_fr = frac_arr[1].reshape(temp.shape).copy()
    ice_fr   = frac_arr[2].reshape(temp.shape).copy()

    for i in range(temp.shape[0]):
        for j in range(temp.shape[1]):
            T = temp[i,j]
            # If already all ice (when T<0) or all ocean (when T>0), skip.
            if (T < 273.15 and ocean_fr[i,j] == 0) or (T > 273.15 and ice_fr[i,j] == 0):
                continue
            diff = T - 273.15
            frac_change = diff * change_per_deg
            new_ice   = ice_fr[i,j]   - frac_change
            new_ocean = ocean_fr[i,j] + frac_change

            # Clamp & conserve at the cell level
            if new_ice < 0.0:
                new_ocean += new_ice; new_ice = 0.0
            if new_ocean < 0.0:
                new_ice += new_ocean; new_ocean = 0.0
            if new_ice > 1.0:
                new_ocean = 0.0; new_ice = 1.0
            if new_ocean > 1.0:
                new_ice = 0.0; new_ocean = 1.0

            ice_fr[i,j]   = new_ice
            ocean_fr[i,j] = new_ocean

    new_frac = np.vstack([land_fr.flatten(), ocean_fr.flatten(), ice_fr.flatten()])
    new_alb = np.array([np.sum(new_frac[:,k]*albedo_arr) for k in range(new_frac.shape[1])]).reshape(temp.shape)
    return new_frac, new_alb

# -----------------------------------------------------------------------------
# Build φ(t, lon, lat) from the eruption list
# -----------------------------------------------------------------------------
def volcanic_phi_field(t_days):
    """
    Returns φ on the 12×6 grid at time t_days (in [0.15, 1]).
    """
    phi = np.ones((12,6))
    LAT, LON = np.meshgrid(0.5*(lat_edges[:-1]+lat_edges[1:]),
                           0.5*(lon_edges[:-1]+lon_edges[1:]), indexing='ij')
    LAT = LAT.T; LON = LON.T
    for e in ERUPTIONS:
        if t_days < e['t0_days']: 
            continue
        A = e['amplitude']; lat0, lon0 = e['lat'], e['lon']
        dt   = t_days - e['t0_days']
        amp  = A * (1.0 - np.exp(-dt/e['tau_rise_days'])) * np.exp(-dt/e['tau_decay_days'])
        grow = 1.0 - np.exp(-dt/e['tau_spread_days'])
        sig_lat = np.deg2rad(e['sigma_lat0_deg'] + (e['sigma_latmax_deg']-e['sigma_lat0_deg'])*grow)
        sig_lon = np.deg2rad(e['sigma_lon0_deg'] + (e['sigma_lonmax_deg']-e['sigma_lon0_deg'])*grow)
        dlat = np.deg2rad(LAT - lat0)
        dlon = np.deg2rad((LON - lon0 + 180.0) % 360.0 - 180.0)
        w = np.exp(-0.5*(dlat/sig_lat)**2) * np.exp(-0.5*(dlon/sig_lon)**2)
        if w.max() > 0: 
            w /= w.max()
        # amplify in the source hemisphere (crude Brewer–Dobson surrogate)
        w *= np.where(np.sign(LAT)==np.sign(lat0), 1.0+e['hemi_bias'], 1.0)
        phi *= (1.0 - amp*w)
    return np.clip(phi, 0.15, 1.0)

# -----------------------------------------------------------------------------
# One EBM step: energy balance + lateral exchange
# -----------------------------------------------------------------------------
def compute_dT(tempK, frac_arr, avg_albedos, phi_grid):
    """
    Compute ∂T/∂t for each cell (K/s) using:
      Shortwave:  geometric_factors * (1 - albedo_sky) * (1 - albedo_cell) * (S0 * φ)
      Longwave:   transmissivity_atm * sigma_B_scaled * (T/100)^4
      Lateral:    symmetric fluxes in latitude & longitude using Tyler’s stencil.
    Then divide by the bulk heat capacity pcZ (per cell).
    """
    # Local radiative tendency
    dT = (geometric_factors * (1 - albedo_sky) * (1 - avg_albedos) * (solar_const * phi_grid)
          - transmissivity_atm * sigma_B_scaled * (tempK/100.0)**4) / pcZ_vals

    # Meridional exchange (between latitude bands)
    dT[:,:-1] += lengths_lat * k_vals_lat * (tempK[:,1:] - tempK[:,:-1]) / (area_earth * area_fractions[:,:-1] * pcZ_vals[:,:-1])
    dT[:,1:]  -= lengths_lat * k_vals_lat * (tempK[:,1:] - tempK[:,:-1]) / (area_earth * area_fractions[:,1:]  * pcZ_vals[:,1:])

    # Zonal exchange (between longitudes), including periodic wrap
    dT[:-1]   += lengths_lon[:-1] * k_vals_lon[:-1] * (tempK[1:] - tempK[:-1]) / (area_earth * area_fractions[:-1] * pcZ_vals[:-1])
    dT[1:]    -= lengths_lon[:-1] * k_vals_lon[:-1] * (tempK[1:] - tempK[:-1]) / (area_earth * area_fractions[1:]  * pcZ_vals[1:])
    dT[0]     -= lengths_lon[-1]  * k_vals_lon[-1]  * (tempK[0] - tempK[-1])  / (area_earth * area_fractions[0]   * pcZ_vals[0])
    dT[-1]    += lengths_lon[-1]  * k_vals_lon[-1]  * (tempK[0] - tempK[-1])  / (area_earth * area_fractions[-1]  * pcZ_vals[-1])

    return dT

# -----------------------------------------------------------------------------
# Integrate forward in time
# -----------------------------------------------------------------------------
dt = DT_DAYS * 24*3600.0                      # seconds per step
nsteps = int(YEARS*365.0/DT_DAYS)             # total number of steps

# Initial temperature and fractions (Tyler’s default)
temp = np.ones((12,6)) * 280.0                # 280 K everywhere
frac_arr   = frac_arr_init.copy()
avg_albedos = avg_albedos_init.copy()

# Storage for the two panels (float32 to save memory)
Temps     = np.zeros((nsteps,12,6), dtype=np.float32)
OceanFrac = np.zeros_like(Temps, dtype=np.float32)
Temps[0] = temp
OceanFrac[0] = frac_arr[1].reshape((12,6))

# Main loop: radiative + diffusive step, clip extreme T, then update sea-ice
for k in range(1, nsteps):
    t_days = k*DT_DAYS
    phi = volcanic_phi_field(t_days)
    dT  = compute_dT(temp, frac_arr, avg_albedos, phi)
    temp = temp + dT*dt
    temp = np.clip(temp, 180.0, 340.0)        # hard guard against numerical blow-ups
    frac_arr, avg_albedos = change_ice_frac(temp, frac_arr, change_per_deg=0.05/365.0)
    Temps[k]     = temp
    OceanFrac[k] = frac_arr[1].reshape((12,6))

# Temperatures in °C for plotting
TempsC = Temps - 273.15

# -----------------------------------------------------------------------------
# Interactive plotting (Mostly GPT)
# -----------------------------------------------------------------------------
extent = [-180, 180, -90, 90]
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(11,12))
plt.subplots_adjust(bottom=0.12, hspace=0.22)

def draw_bg(ax):
    """Draw Natural Earth basemap under the rasters (once)."""
    if not USE_BASEMAP: return
    path = _first_existing(NE1_PATHS)
    if path is None or basemap_rgb is None: return
    ax.imshow(basemap_rgb/255.0 if basemap_rgb.dtype!=np.float32 else basemap_rgb,
              extent=extent, origin='upper', zorder=0)

def draw_panel(ax, data, title, vmin, vmax, cmap, label):
    """Helper: place a raster on top of the basemap with gridlines + colorbar."""
    draw_bg(ax)
    img = ax.imshow(data.T, extent=extent, origin='lower', vmin=vmin, vmax=vmax,
                    cmap=cmap, alpha=0.78, zorder=1)
    for x in lon_edges: ax.axvline(x, color='white', lw=0.6, ls='-', alpha=0.6, zorder=2)
    for y in lat_edges: ax.axhline(y, color='white', lw=0.6, ls='-', alpha=0.6, zorder=2)
    ax.set_xticks(np.arange(-180,181,60)); ax.set_yticks(np.arange(-90,91,30))
    ax.set_xlim(-180,180); ax.set_ylim(-90,90)
    ax.set_title(title)
    cb = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.02); cb.set_label(label)
    return img

# First frame for each panel
imT = draw_panel(ax1, TempsC[0], "Surface Temperature (°C) — Day 0",
                 vmin=float(np.nanmin(TempsC)), vmax=float(np.nanmax(TempsC)),
                 cmap='coolwarm', label="Temperature (°C)")
imF = draw_panel(ax2, OceanFrac[0], "Ocean Fraction — Day 0",
                 vmin=0.0, vmax=1.0, cmap='YlGnBu', label="Ocean Fraction")

# Slider + play/pause controls (shared for both panels)
ax_slider = plt.axes([0.12, 0.05, 0.70, 0.03]); slider = Slider(ax_slider, 'Day', 0, nsteps-1, valinit=0, valfmt='%d')
ax_play  = plt.axes([0.84, 0.045, 0.06, 0.035]); btn_play  = Button(ax_play,  '▶ Play')
ax_pause = plt.axes([0.91, 0.045, 0.06, 0.035]); btn_pause = Button(ax_pause, '❚❚ Pause')
_running = {'on': False}

def _update(val):
    """Slider callback: update both rasters and titles."""
    k = int(slider.val)
    imT.set_data(TempsC[k].T); imF.set_data(OceanFrac[k].T)
    ax1.set_title(f"Surface Temperature (°C) — Day {k}")
    ax2.set_title(f"Ocean Fraction — Day {k}")
    fig.canvas.draw_idle()

slider.on_changed(_update)

def _animate(event):
    """Play: advance the slider forward in real time."""
    _running['on'] = True
    k = int(slider.val)
    while _running['on'] and k < nsteps-1:
        k += 1; slider.set_val(k); plt.pause(0.02)

def _stop(event):
    """Pause the animation loop."""
    _running['on'] = False

btn_play.on_clicked(_animate); btn_pause.on_clicked(_stop)

plt.show()
