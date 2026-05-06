
#------------------------
# Imports
#------------------------
import math
import string
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Rectangle
from matplotlib.lines import Line2D
from PIL import Image, ImageDraw

#------------------------
# general utilities
#------------------------
def saveoutputs(df: pd.DataFrame, path: str, index: bool = False):
    """
    Save a DataFrame to CSV.

    Parameters
   ------
    df : pd.DataFrame
        Data to save.
    path : str
        Output file path.
    index : bool
        Whether to write row index.
    """
    df.to_csv(path, index=index)

#------------------------
# geometry utilities
#------------------------
def rotationmatrix(thetarad: float) -> np.ndarray:
    c, s = np.cos(thetarad), np.sin(thetarad)
    return np.array([[c, -s],
                     [s,  c]], dtype=float)

def polygonarea(coords: np.ndarray) -> float:
    """
    Shoelace area for a simple polygon (coords: (N,2), first point need not be repeated).
    """
    if coords.shape[0] < 3:
        return 0.0
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

def cross2d(o, a, b) -> float:
    return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

def convexhull(points: np.ndarray) -> np.ndarray:
    """
    Andrew's monotone chain convex hull. Returns CCW vertices without repeating the first point.
    """
    pts = np.unique(points, axis=0)
    if pts.shape[0] <= 1:
        return pts.copy()
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross2d(lower[-2], lower[-1], tuple(p)) <= 0:
            lower.pop()
        lower.append(tuple(p))

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross2d(upper[-2], upper[-1], tuple(p)) <= 0:
            upper.pop()
        upper.append(tuple(p))

    hull = lower[:-1] + upper[:-1]
    return np.array(hull, dtype=float)

def pointinconvexpolygon(p: tuple[float, float], poly: np.ndarray) -> bool:
    """
    Point-in-polygon and point-to-polygon
    True if p is inside or on a convex polygon 'poly' (assumed CCW). Vectorized over edges.
    """
    m = poly.shape[0]
    if m == 0:
        return False
    if m == 1:
        return (abs(p[0] - poly[0, 0]) < 1e-15) and (abs(p[1] - poly[0, 1]) < 1e-15)
    if m == 2:
        ax, ay = poly[0]
        bx, by = poly[1]
        cross = (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)
        if abs(cross) > 1e-12:
            return False
        dot = (p[0] - ax) * (bx - ax) + (p[1] - ay) * (by - ay)
        seglen2 = (bx - ax) ** 2 + (by - ay) ** 2
        return 0.0 <= dot <= seglen2

    p = np.array(p, dtype=float)
    v0 = poly
    v1 = np.roll(poly, -1, axis=0)
    cp = (v1[:, 0] - v0[:, 0]) * (p[1] - v0[:, 1]) - (v1[:, 1] - v0[:, 1]) * (p[0] - v0[:, 0])
    return np.all(cp >= -1e-12)

def pointtosegmentdistance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    px, py = p
    ax, ay = a
    bx, by = b
    abx, aby = (bx - ax), (by - ay)
    apx, apy = (px - ax), (py - ay)
    ab2 = abx * abx + aby * aby
    if ab2 == 0.0:
        return math.hypot(apx, apy)
    t = (apx * abx + apy * aby) / ab2
    t = max(0.0, min(1.0, t))
    qx = ax + t * abx
    qy = ay + t * aby
    return math.hypot(px - qx, py - qy)

def distancepointtopolygon(p: tuple[float, float], poly: np.ndarray) -> float:
    """
    Minimum Euclidean distance from p to polygon boundary (handles degenerate 1- or 2-vertex cases).
    Vectorized across segments for speed.
    """
    m = poly.shape[0]
    if m == 0:
        return float('nan')
    if m == 1:
        return math.hypot(p[0] - poly[0, 0], p[1] - poly[0, 1])
    if m == 2:
        return pointtosegmentdistance(p, tuple(poly[0]), tuple(poly[1]))

    p = np.array(p, dtype=float)
    v0 = poly
    v1 = np.roll(poly, -1, axis=0)
    ab = v1 - v0
    ap = p - v0
    ab2 = np.einsum('ij,ij->i', ab, ab)
    denom = np.where(ab2 == 0.0, 1.0, ab2)
    t = np.clip((ap[:, 0] * ab[:, 0] + ap[:, 1] * ab[:, 1]) / denom, 0.0, 1.0)
    q = v0 + ab * t[:, None]
    d = np.hypot(q[:, 0] - p[0], q[:, 1] - p[1])
    return float(d.min())



#------------------------
# leave-k-out on observed vents
#------------------------

def withholdkensemble(
    points: np.ndarray,
    k: int | list[int] = 2,
    nsims: int | None = 1000,
    baseseed: int | None = 123,
    replacement: bool = True,
    cycles: int = 1
) -> pd.DataFrame:
    """
    Many leave-k-out trials; long-form result (one row per withheld vent).

    Parameters
   ------
    points : ndarray (N, 2)
        Vent coordinates.
    k : int or list[int]
        Number(s) of vents to withhold.
    nsims : int
        Number of simulations (for replacement=True).
    replacement : bool
        Whether to sample k-sets with replacement.
    cycles : int
        Number of full combination cycles (if replacement=False).

    Returns
   ---
    DataFrame
        Long-form results with kleftout column.
    """

    # normalize k to list
    if isinstance(k, (int, np.integer)):
        ks = [int(k)]
    else:
        ks = [int(kk) for kk in k]

    n = int(points.shape[0])
    rng = np.random.default_rng(baseseed)
    allrows = []
    simid = 0  # global simid across all k

    for kcur in ks:
        if kcur < 1 or kcur >= n:
            raise ValueError("k must be between 1 and N-1")

        if replacement:
            if nsims is None or nsims <= 0:
                raise ValueError("nsims must be positive when replacement=True")

            for _ in range(nsims):
                simid += 1
                chosen = rng.choice(n, size=kcur, replace=False)
                mask = np.ones(n, dtype=bool)
                mask[chosen] = False
                kept = points[mask, :]

                if kept.shape[0] >= 3:
                    hull = convexhull(kept)
                else:
                    hull = kept.copy()

                for idx in chosen:
                    p = tuple(points[idx, :])
                    inside = pointinconvexpolygon(p, hull)
                    distout = 0.0 if inside else float(distancepointtopolygon(p, hull))

                    allrows.append({
                        'simid': simid,
                        'withheldindex': int(idx),
                        'inside': bool(inside),
                        'distout': distout,
                        'kleftout': int(kcur)
                    })

        else:
            from itertools import combinations

            combos = list(combinations(range(n), kcur))
            for _ in range(cycles):
                order = range(len(combos)) 
                for j in order:
                    simid += 1
                    chosen = np.array(combos[j], dtype=int)
                    mask = np.ones(n, dtype=bool)
                    mask[chosen] = False
                    kept = points[mask, :]

                    if kept.shape[0] >= 3:
                        hull = convexhull(kept)
                    else:
                        hull = kept.copy()

                    for idx in chosen:
                        p = tuple(points[idx, :])
                        inside = pointinconvexpolygon(p, hull)
                        distout = 0.0 if inside else float(distancepointtopolygon(p, hull))

                        allrows.append({
                            'simid': simid,
                            'withheldindex': int(idx),
                            'inside': bool(inside),
                            'distout': distout,
                            'kleftout': int(kcur)
                        })

    return pd.DataFrame(
        allrows,
        columns=['simid', 'withheldindex', 'inside', 'distout', 'kleftout']
    )


def summarizewithholdk(
    results: pd.DataFrame,
    target_percentile: float | list[float] | None = None,
    target_buffer_m: float | list[float] | None = None
) -> dict:
    """
    Summarize leave-k-out results.

    Provide EXACTLY ONE of:
      - target_percentile (float or list of floats, e.g. 0.99 or [0.95, 0.99])
      - target_buffer_m  (float or list of floats, metres)

    Returns:
      summaries[target][k] = {
          'target_percentile': ...,
          'target_buffer_m': ...,
          'p_inside': ...,
          'nventsassessed': ...,
          'ntrials': ...
      }
    """

    # --- validation ---
    if (target_percentile is None) == (target_buffer_m is None):
        raise ValueError("Provide exactly one of target_percentile or target_buffer_m")

    if results.empty:
        return {}

    # normalize targets
    if target_percentile is not None:
        if isinstance(target_percentile, (int, float)):
            targets = [float(target_percentile)]
        else:
            targets = [float(p) for p in target_percentile]
        mode = 'percentile'
    else:
        if isinstance(target_buffer_m, (int, float)):
            targets = [float(target_buffer_m)]
        else:
            targets = [float(b) for b in target_buffer_m]
        mode = 'buffer'

    summaries = {}

    # loop over targets and k
    for t in targets:
        summaries[t] = {}

        for k, g in results.groupby('kleftout'):
            inside = g['inside'].astype(bool).to_numpy()
            dist = np.maximum(0.0, g['distout'].astype(float).to_numpy())

            nvents = int(dist.size)
            ntrials = g['simid'].nunique()
            p_inside = inside.mean() if nvents else np.nan

            if mode == 'percentile':
                buf = float(np.nanpercentile(dist, 100.0 * t)) if nvents else np.nan
                pct = float(t)
            else:
                buf = float(t)
                pct = float((dist <= buf).mean()) if nvents else np.nan

            summaries[t][int(k)] = {
                'target_percentile': pct,
                'target_buffer_m': buf,
                'p_inside': p_inside,
                'nventsassessed': nvents,
                'ntrials': int(ntrials)
            }

    return summaries


#------------------------
# synthetic sampling
#------------------------

def samplepointsinellipse(n: int,
                          center: tuple[float, float],
                          a: float,
                          b: float,
                          rotationdeg: float,
                          rng: np.random.Generator) -> np.ndarray:
    """
    Uniformly sample n points inside a rotated ellipse with semi-axes a (x) and b (y).
    """
    theta = rng.uniform(0, 2 * np.pi, size=n)
    r = np.sqrt(rng.uniform(0, 1, size=n))
    ux = r * np.cos(theta)
    uy = r * np.sin(theta)
    pts = np.vstack([ux * a, uy * b]).T
    R = rotationmatrix(np.deg2rad(rotationdeg))
    pts = pts @ R.T
    pts[:, 0] += center[0]
    pts[:, 1] += center[1]
    return pts

def trueellipsearea(a: float, b: float) -> float:
    return math.pi * a * b

def sequentialhullinfo(
    vents: pd.DataFrame,
    xcol: str = 'easting',
    ycol: str = 'northing',
    ordercol: str = 'index',
    startrect: tuple[float, float] = (28884, 16481),
    center: tuple[float, float] = (76614, 71891),
    rotationdeg: float = 90 - 2.6,
    buffers: float | list[float] | None = None,   # metres
    hullareacol: str = 'hullareakm2',
    arearatiocol: str = 'arearatio',
    insidecol: str = 'ventinprevhull',
    distcol: str = 'distoutsidekm',
    savepath: str | None = None
) -> pd.DataFrame:

    df = vents.copy()
    df[ordercol] = pd.to_numeric(df[ordercol], errors='coerce')
    df[hullareacol] = np.nan
    df[arearatiocol] = np.nan
    if buffers is None:
        buffers = []
    elif isinstance(buffers, (int, float)):
        buffers = [buffers]
    buffer_info = [] 
    for buf in buffers:
        if buf > 0:
            buf_m = int(round(buf))
            col = f'bufferedarearatio_{buf_m}m'
            df[col] = np.nan
            buffer_info.append((buf, buf_m, col))
    df[insidecol] = pd.Series(pd.NA, dtype="boolean")
    df[distcol] = np.nan

    # starting ellipse area (m²)
    rectw, recth = startrect
    a = rectw / 2.0
    b = recth / 2.0
    startarea = trueellipsearea(a, b)

    # precompute buffered reference areas
    startareas_buf = {
        col: trueellipsearea(a + buf, b + buf)
        for buf, _, col in buffer_info
    }

    # valid vents in eruption order
    valid = df.dropna(subset=[ordercol]).sort_values(ordercol)
    points = valid[[xcol, ycol]].to_numpy(float)
    row_indices = valid.index.to_numpy()

    prevhull = None

    for i in range(len(points)):
        t = i + 1
        p = tuple(points[i])

        # distance/inside previous hull
        if t <= 2:
            df.loc[row_indices[i], insidecol] = np.nan
            df.loc[row_indices[i], distcol] = np.nan
        else:
            inside = pointinconvexpolygon(p, prevhull)
            df.loc[row_indices[i], insidecol] = bool(inside)
            df.loc[row_indices[i], distcol] = (
                0.0 if inside else distancepointtopolygon(p, prevhull) / 1000.0
            )

        # current hull area
        if t < 3:
            hullarea = 0.0
            prevhull = points[:t, :].copy()
        else:
            hull = convexhull(points[:t, :])
            hullarea = polygonarea(hull)
            prevhull = hull

        # unbuffered area ratio
        df.loc[row_indices[i], hullareacol] = hullarea
        df.loc[row_indices[i], arearatiocol] = (
            hullarea / startarea if startarea > 0 else np.nan
        )

        # buffered area ratios
        for _, _, col in buffer_info:
            df.loc[row_indices[i], col] = (
                hullarea / startareas_buf[col]
                if startareas_buf[col] > 0 else np.nan
            )

    # save if requested
    if savepath is not None:
        df.to_csv(savepath, index=False)

    return df

def simulateone(n_eruptions: int,
                coordsystem: str = 'cartesian',
                center: tuple[float, float] = (0.0, 0.0),
                a: float | None = None,
                b: float | None = None,
                startrect: tuple[float, float] | None = None,
                rotationdeg: float = 0.0,
                unitscale: float = 1.0,
                baseseed: int | None = None) -> pd.DataFrame:
    """
    Simulate eruptions inside a rotated ellipse and track:
      - hull area and hull/ellipse area ratio at each step
      - distance to previous vent (distprev)
      - inside-previous-hull flag and distance to previous hull if outside
    """
    if coordsystem.lower() != 'cartesian':
        raise ValueError("only 'cartesian' is implemented")
    if (a is None or b is None):
        if startrect is None:
            raise ValueError("provide either a,b or startrect=(width,height)")
        rectw, recth = startrect
        a = rectw / 2.0
        b = recth / 2.0

    rng = np.random.default_rng(baseseed)
    pts = samplepointsinellipse(n_eruptions, center, a, b, rotationdeg, rng)

    if unitscale != 1.0:
        pts = pts * unitscale
        a_scaled = a * unitscale
        b_scaled = b * unitscale
    else:
        a_scaled, b_scaled = a, b

    truearea = trueellipsearea(a_scaled, b_scaled)

    hullareas = np.zeros(n_eruptions, dtype=float)
    arearatios = np.zeros(n_eruptions, dtype=float)
    distprev = np.full(n_eruptions, np.nan, dtype=float)
    withinprevhull = np.full(n_eruptions, np.nan, dtype=object)
    disttoprevhull = np.full(n_eruptions, np.nan, dtype=float)

    prevhull = None  # hull for points up to t-1

    for t in range(1, n_eruptions + 1):
        sub = pts[:t, :]

        # inside/outside vs previous hull
        if t == 1:
            withinprevhull[t-1] = np.nan
            disttoprevhull[t-1] = np.nan
        elif t == 2:
            pnew = (pts[t-1, 0], pts[t-1, 1])
            prevsub = pts[:t-1, :]
            if prevsub.shape[0] == 1:
                inside = False
                dph = float(math.hypot(pnew[0] - prevsub[0, 0], pnew[1] - prevsub[0, 1]))
                withinprevhull[t-1] = bool(inside)
                disttoprevhull[t-1] = dph
                prevhull = prevsub.copy()  # segment prep for next step
            else:
                prevhull = prevsub.copy()
                inside = pointinconvexpolygon(pnew, prevhull)
                withinprevhull[t-1] = bool(inside)
                disttoprevhull[t-1] = 0.0 if inside else distancepointtopolygon(pnew, prevhull)
        else:
            pnew = (pts[t-1, 0], pts[t-1, 1])
            inside = pointinconvexpolygon(pnew, prevhull)
            withinprevhull[t-1] = bool(inside)
            disttoprevhull[t-1] = 0.0 if inside else distancepointtopolygon(pnew, prevhull)

        # current hull for area (and carry forward)
        if sub.shape[0] >= 3:
            currhull = convexhull(sub)
            hullarea = polygonarea(currhull)
            prevhull = currhull
        else:
            hullarea = 0.0

        hullareas[t-1] = hullarea
        arearatios[t-1] = hullarea / truearea if truearea > 0 else np.nan

        if t >= 2:
            dx = pts[t-1, 0] - pts[t-2, 0]
            dy = pts[t-1, 1] - pts[t-2, 1]
            distprev[t-1] = math.hypot(dx, dy)

    return pd.DataFrame({
        't': np.arange(1, n_eruptions + 1, dtype=int),
        'x': pts[:, 0],
        'y': pts[:, 1],
        'hullarea': hullareas,
        'arearatio': arearatios,
        'distprev': distprev,
        'withinprevhull': withinprevhull,
        'disttoprevhull': disttoprevhull
    })

def runensemble(n_sims: int,
                n_eruptions: int,
                coordsystem: str = 'cartesian',
                center: tuple[float, float] = (0.0, 0.0),
                a: float | None = None,
                b: float | None = None,
                startrect: tuple[float, float] | None = None,
                rotationdeg: float = 0.0,
                unitscale: float = 1.0,
                baseseed: int | None = 42) -> pd.DataFrame:
    """
    Stack multiple simulateone runs; returns long DataFrame with simid column.
    """
    allruns = []
    for simid in range(1, n_sims + 1):
        seed = None if baseseed is None else baseseed + simid * 1009
        df = simulateone(
            n_eruptions=n_eruptions,
            coordsystem=coordsystem,
            center=center,
            a=a, b=b,
            startrect=startrect,
            rotationdeg=rotationdeg,
            unitscale=unitscale,
            baseseed=seed
        )
        df.insert(0, 'simid', simid)
        allruns.append(df)
    return pd.concat(allruns, ignore_index=True)

def computepercentiles(ensemble: pd.DataFrame,
                       percentiles: list[float] = [2.5, 16, 50, 84, 97.5],
                       valuecols: list[str] = ['hullarea', 'arearatio', 'distprev']) -> pd.DataFrame:
    """
    Time-wise percentiles for selected metrics across simulations.
    """
    pctvals = [p / 100.0 for p in percentiles]
    out = []
    for t, g in ensemble.groupby('t', as_index=False):
        for col in valuecols:
            q = g[col].quantile(pctvals)
            for p, val in zip(percentiles, q.values):
                out.append({'t': t, 'metric': col, 'percentile': p, 'value': val})
    return pd.DataFrame(out)


#------------------------
# plotting helpers
#------------------------

def computehulldensitygrid(ensemble: pd.DataFrame,
                           eruptionnum: int,
                           center: tuple[float, float],
                           a: float | None = None,
                           b: float | None = None,
                           startrect: tuple[float, float] | None = None,
                           rotationdeg: float = 0.0,
                           mapbuffer: float = 2.0,
                           gridsize: tuple[int, int] = (300, 300)) -> tuple[np.ndarray, tuple[float,float,float,float]]:
    """
    Build a coverage-frequency grid for hulls at a given eruption number across all simulations.
    Returns:
      density  : (ny, nx) float array in [0,1], where 1 = cell is inside the hull for all sims
      extentkm : (minx_km, maxx_km, miny_km, maxy_km) for plotting
    Notes:
      - Ensemble coordinates are metres; the grid is defined in km for plotting.
      - Uses Pillow to raster-fill each hull into an accumulator (fast and scalable).
    """
    import numpy as np
    from PIL import Image, ImageDraw

    # resolve ellipse semi-axes
    if (a is None or b is None):
        if startrect is None:
            raise ValueError("computehulldensitygrid needs a,b or startrect")
        rectw, recth = startrect
        a = rectw / 2.0
        b = recth / 2.0

    cx_m, cy_m = center
    theta = np.deg2rad(rotationdeg)

    # ellipse AABB half-extents (metres) for map extent
    hx_m = float(np.hypot(a * np.cos(theta), b * np.sin(theta)))
    hy_m = float(np.hypot(a * np.sin(theta), b * np.cos(theta)))

    # ellipse-only (no buffer) ticks in km
    minx_km_ell = (cx_m - hx_m) / 1000.0
    maxx_km_ell = (cx_m + hx_m) / 1000.0
    miny_km_ell = (cy_m - hy_m) / 1000.0
    maxy_km_ell = (cy_m + hy_m) / 1000.0

    # view with asymmetric buffer (same convention as your plotsummary)
    left_buf   = 1.5 * mapbuffer
    right_buf  = 1.0 * mapbuffer
    bottom_buf = 1.0 * mapbuffer
    top_buf    = 1.5 * mapbuffer

    minx_km = minx_km_ell - left_buf
    maxx_km = maxx_km_ell + right_buf
    miny_km = miny_km_ell - bottom_buf
    maxy_km = maxy_km_ell + top_buf

    nx, ny = int(gridsize[0]), int(gridsize[1])
    acc = np.zeros((ny, nx), dtype=np.uint32)

    nsims = int(ensemble['simid'].nunique())
    if nsims == 0:
        return acc.astype(float), (minx_km, maxx_km, miny_km, maxy_km)

    # helper to map km -> pixel coords
    def km_to_px(xkm, ykm):
        # x: [minx_km, maxx_km] -> [0, nx-1]
        xpx = (xkm - minx_km) / max(1e-12, (maxx_km - minx_km)) * (nx - 1)
        # y: [miny_km, maxy_km] -> [ny-1, 0] (image origin top-left)
        ypx = (1.0 - (ykm - miny_km) / max(1e-12, (maxy_km - miny_km))) * (ny - 1)
        return xpx, ypx

    # draw each hull as a filled polygon on a mask, then accumulate
    for sid, grp in ensemble.groupby('simid', sort=True):
        sub = grp[grp['t'] <= eruptionnum]
        if len(sub) < 3:
            continue
        pts_m = sub[['x', 'y']].to_numpy()
        hull_m = convexhull(pts_m)
        # convert to km
        xkm = hull_m[:, 0] / 1000.0
        ykm = hull_m[:, 1] / 1000.0
        # to pixel coords
        xpx, ypx = km_to_px(xkm, ykm)
        poly_px = list(map(lambda xy: (float(np.clip(xy[0], 0, nx-1)),
                                       float(np.clip(xy[1], 0, ny-1))),
                           zip(xpx, ypx)))

        img = Image.new('L', (nx, ny), 0)
        draw = ImageDraw.Draw(img)
        if len(poly_px) >= 3:
            draw.polygon(poly_px, outline=None, fill=1)
        else:
            # degenerate hull (shouldn't happen for >=3), skip
            continue
        acc += np.asarray(img, dtype=np.uint32)

    density = acc.astype(np.float64) / float(nsims)
    return density, (minx_km, maxx_km, miny_km, maxy_km)

def plotspatial(
    ax,
    ensemble: pd.DataFrame,
    center: tuple[float, float],
    a: float,
    b: float,
    startrect: tuple[float, float] | None = None,
    rotationdeg: float = 0.0,
    eruptionnum: int = 55,
    mapbuffer: float = 2.0,
    ncolor: int = 10,
    hullalpha: float = 0.001,
):

    cx_m, cy_m = center
    theta = np.deg2rad(rotationdeg)

    # Ellipse AABB half extents (m)
    hx_m = float(np.hypot(a * np.cos(theta), b * np.sin(theta)))
    hy_m = float(np.hypot(a * np.sin(theta), b * np.cos(theta)))

    # Ellipse bounds [km]
    minx_km_ell = (cx_m - hx_m) / 1000.0
    maxx_km_ell = (cx_m + hx_m) / 1000.0
    miny_km_ell = (cy_m - hy_m) / 1000.0
    maxy_km_ell = (cy_m + hy_m) / 1000.0

    # Plot extent with buffer
    left_buf   = 1.5 * mapbuffer
    right_buf  = 1.0 * mapbuffer
    bottom_buf = 1.0 * mapbuffer
    top_buf    = 1.5 * mapbuffer

    minx_km = minx_km_ell - left_buf
    maxx_km = maxx_km_ell + right_buf
    miny_km = miny_km_ell - bottom_buf
    maxy_km = maxy_km_ell + top_buf

    # Background (outside ellipse)
    rect = Rectangle(
        (minx_km, miny_km),
        maxx_km - minx_km,
        maxy_km - miny_km,
        facecolor='#EAEAEA',
        edgecolor='none',
        zorder=0
    )
    ax.add_patch(rect)

    # Starting ellipse
    cx_km, cy_km = cx_m / 1000.0, cy_m / 1000.0

    ell_fill = Ellipse(
        xy=(cx_km, cy_km),
        width=(2 * a) / 1000.0,
        height=(2 * b) / 1000.0,
        angle=rotationdeg,
        facecolor=ax.get_facecolor(),
        edgecolor='none',
        zorder=1
    )
    ax.add_patch(ell_fill)

    ell_edge = Ellipse(
        xy=(cx_km, cy_km),
        width=(2 * a) / 1000.0,
        height=(2 * b) / 1000.0,
        angle=rotationdeg,
        facecolor='none',
        edgecolor='black',
        linewidth=1.6,
        zorder=3
    )
    ax.add_patch(ell_edge)

    # Hull density grid
    density, extent = computehulldensitygrid(
        ensemble=ensemble,
        eruptionnum=eruptionnum,
        center=center,
        a=a,
        b=b,
        startrect=startrect,
        rotationdeg=rotationdeg,
        mapbuffer=mapbuffer,
        gridsize=(300, 300)
    )

    density = np.flipud(density) * 100.0
    density = np.ma.masked_where(density <= 0.0, density)

    cmap = plt.get_cmap('BuPu', ncolor)

    im = ax.imshow(
        density,
        extent=extent,
        cmap=cmap,
        origin='lower',
        vmin=1e-9,
        vmax=100,
        interpolation='nearest',
        zorder=2
    )

    # Colorbar (top)
    cb = ax.figure.colorbar(
        im, ax=ax,
        orientation='horizontal',
        location='top',
        fraction=0.05,
        pad=0.05,
        shrink=0.75
    )
    cb.set_ticks(np.arange(0, 101, ncolor))
    nsims = ensemble['simid'].nunique()
    cb.set_label(f'Density of {eruptionnum}th hull across {nsims} simulations [%]')

    # Axes formatting
    ax.set_xlim(minx_km, maxx_km)
    ax.set_ylim(miny_km, maxy_km)
    ax.set_aspect('equal', adjustable='box')

    ax.set_xlabel('Easting [km]')
    ax.set_ylabel('Northing [km]')

def plotarearatio(
    ax,
    ensemble,
    percentiles,
    observed=None,
    bufferedarearatiocol: list[str] | None = None,
    synthetic_color='#018571',
    observed_color='#a6611a'
):

    # --- synthetic percentiles ---
    p = percentiles[percentiles['metric'] == 'arearatio']
    piv = p.pivot(index='t', columns='percentile', values='value')

    # start at eruption 3
    piv = piv[piv.index >= 3]

    if {2.5, 97.5}.issubset(piv.columns):
        ax.fill_between(
            piv.index, piv[2.5], piv[97.5],
            color=synthetic_color, alpha=0.12,
            label='Synthetic 2.5–97.5%'
        )
    if {5, 95}.issubset(piv.columns):
        ax.fill_between(
            piv.index, piv[5], piv[95],
            color=synthetic_color, alpha=0.12,
            label='Synthetic 5–95%'
        )
    if {16, 84}.issubset(piv.columns):
        ax.fill_between(
            piv.index, piv[16], piv[84],
            color=synthetic_color, alpha=0.25,
            label='Synthetic 16–84%'
        )
    if 50 in piv.columns:
        ax.plot(
            piv.index, piv[50],
            color=synthetic_color, lw=1.5,
            label='Synthetic median'
        )

    # --- observed (unbuffered) ---
    if observed is not None and 'arearatio' in observed.columns:
        obs = observed.sort_values('index')

        # start at eruption 3
        obs = obs[obs['index'] >= 3]

        ax.plot(
            obs['index'], obs['arearatio'],
            color=observed_color, lw=1,
            label='AVF (observed)'
        )

        # --- observed (buffered) ---
        if bufferedarearatiocol is not None:
            for col in bufferedarearatiocol:
                if col in obs.columns:
                    buf_label = col.replace('bufferedarearatio_', '')
                    ax.plot(
                        obs['index'], obs[col],
                        color=observed_color,
                        lw=1,
                        ls='--',
                        label=f'AVF (with {buf_label} buffer)'
                    )

    ax.set_xlabel('Eruption count [t]')
    ax.set_ylabel('Area ratio')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    ylow, _ = ax.get_ylim()
    ax.set_ylim(ylow, 1.0)

def plotprctoutside(
    ax,
    ensemble,
    synthetic_color='#018571'
):
    def frac_outside(s):
        s = s.dropna()
        return np.nan if len(s) == 0 else (s == False).mean() * 100.0

    pct_out = ensemble.groupby('t')['withinprevhull'].apply(frac_outside)
    pct_out = pct_out[pct_out.index >= 3]

    ax.plot(
        pct_out.index, pct_out.values,
        color=synthetic_color, lw=1.5,
        label='Synthetic'
    )

    ax.set_ylim(0, 100)
    ax.set_xlabel('Eruption count [t]')
    ax.set_ylabel('Vent outside hull [%]')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)

def plotprctinside(
    ax,
    ensemble,
    synthetic_color='#018571'
):
    def frac_outside(s):
        s = s.dropna()
        return np.nan if len(s) == 0 else (s == False).mean() * 100.0

    pct_out = ensemble.groupby('t')['withinprevhull'].apply(frac_outside)
    pct_out = pct_out[pct_out.index >= 3]

    ax.plot(
        pct_out.index, 100-pct_out.values,
        color=synthetic_color, lw=1.5,
        label='Synthetic'
    )

    ax.set_ylim(None, 100)
    ax.set_xlabel('Eruption count [t]')
    ax.set_ylabel('Vent within previous hull [%]')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)


def plotdistanceoutside(
    ax,
    ensemble,
    observed=None,
    qs=[0.9, 0.95, 0.975, 0.99, 1.0],
    synthetic_color='#018571',
    observed_color='#a6611a'
):
    """
    Distance outside hull vs eruption number with nested percentile shading.

    - Envelopes extend from 0 to each quantile in `qs`
    - Widest envelope drawn first
    - Alphas increase inward
    - Sum of alphas == 1 (innermost cumulative alpha == 1)
    - Plot starts at t >= 3
    """

    # sort widest -> narrowest
    qs = sorted(qs, reverse=True)
    N = len(qs)

    # outside vents only
    outside = ensemble.loc[
        ensemble['withinprevhull'] == False,
        ['t', 'disttoprevhull']
    ]

    if outside.empty:
        return

    # quantiles by eruption (km)
    qtab = (
        outside
        .groupby('t')['disttoprevhull']
        .quantile(qs)
        .unstack()
        .sort_index()
    ) / 1000.0

    # start at eruption >= 3
    qtab = qtab[qtab.index >= 3]
    tvals = qtab.index

    # alpha ladder that sums to 1
    denom = N * (N + 1) / 2
    alphas = [(i + 1) / denom for i in range(N)]

    # draw envelopes
    for q, alpha in zip(qs, alphas):
        pct = int(round(q * 100))
        ax.fill_between(
            tvals,
            0,
            qtab[q],
            color=synthetic_color,
            alpha=alpha,
            edgecolor='k',
            linewidth=0.5,
            label=f'0–{pct}%'
        )

    # observed points (outside only)
    if observed is not None and 'distoutsidekm' in observed.columns:
        obs_out = observed.loc[
            (observed['distoutsidekm'] > 0.0) &
            (observed['index'] >= 3)
        ]

        if not obs_out.empty:
            ax.scatter(
                obs_out['index'],
                obs_out['distoutsidekm'],
                color=observed_color,
                s=14,
                zorder=10,
                label='Observed AVF'
            )

    ax.set_xlabel('Eruption count (t)')
    ax.set_ylabel('Distance outside hull (km)')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)


def summaryplots(
    ensemble: pd.DataFrame,
    percentiles: pd.DataFrame,
    observed=None,
    boundingellipse=None,
    eruptionnum: int = 55,
    ncolor: int = 10,
    hullalpha: float = 0.001,
    figures: list[tuple[str, list[str]]] = [('summary', ['arearatio'])],
    bufferedarearatiocol: list[str] | None = None,
    synthetic_color: str = '#018571',
    observed_color: str = '#a6611a',
    figsize_panel: tuple[float, float] = (4, 3),
    savefigs: bool = False
) -> list:

    figs = []
    font = {"family": "Arial", "weight": "normal", "size": 8}
    plt.rc("font", **font)

    for fidx, (figname, panels) in enumerate(figures):
        n = len(panels)

        fig, axes = plt.subplots(
            1, n,
            figsize=(figsize_panel[0] * n, figsize_panel[1]),
            sharex=True
        )

        if n == 1:
            axes = [axes]

        for i, (ax, pname) in enumerate(zip(axes, panels)):

            #------ Spatial panel------
            if pname == 'spatial':
                if boundingellipse is None:
                    raise ValueError("Panel 'spatial' requires boundingellipse")

                plotspatial(
                    ax=ax,
                    ensemble=ensemble,
                    center=boundingellipse['center'],
                    a=boundingellipse['a'],
                    b=boundingellipse['b'],
                    rotationdeg=boundingellipse['rotationdeg'],
                    eruptionnum=eruptionnum,
                    ncolor=ncolor,
                    hullalpha=hullalpha
                )

            #------ Area ratio------
            elif pname == 'arearatio':
                plotarearatio(
                    ax=ax,
                    ensemble=ensemble,
                    percentiles=percentiles,
                    observed=observed,
                    bufferedarearatiocol=bufferedarearatiocol,
                    synthetic_color=synthetic_color,
                    observed_color=observed_color
                )

            #------ % outside------
            elif pname == 'prctoutside':
                plotprctoutside(
                    ax=ax,
                    ensemble=ensemble,
                    synthetic_color=synthetic_color,
                )

            #------ % inside------
            elif pname == 'prctinside':
                plotprctinside(
                    ax=ax,
                    ensemble=ensemble,
                    synthetic_color=synthetic_color,
                )

            #------ Distance outside------
            elif pname == 'distanceoutside':
                plotdistanceoutside(
                    ax=ax,
                    ensemble=ensemble,
                    observed=observed,
                    synthetic_color=synthetic_color,
                    observed_color=observed_color
                )

            else:
                raise ValueError(f"Unknown panel name: '{pname}'")

            # Panel lettering (only for multi-panel figures)
            if n > 1:
                ax.text(
                    0.01, 0.98,
                    string.ascii_uppercase[i],
                    transform=ax.transAxes,
                    ha='left', va='top'
                )

        figs.append(fig)

        #- optional save-
        if savefigs:
            fig.savefig(
                f"./Figures/Fig_{figname}.png",
                dpi=200,
                bbox_inches='tight'
            )

        #- always show-
        plt.show()

    return figs


def ensembleplots(
    ensembles: dict,
    observed=None,
    percentile_levels=(2.5, 16, 50, 84, 97.5, 100),
    mode: str = "overlay",   # "overlay" or "separate"
    observed_color: str = "#a6611a",
    figsize_overlay=(12, 4),
    figsize_separate=(12, 10),
):
    """
    Plot ensemble diagnostics for multiple starting-ellipse configurations.
    """

    # font
    font = {"family": "Arial", "weight": "normal", "size": 8}
    plt.rc("font", **font)

    # fixed colours by buffer (m)
    color_by_buffer = {
        0: "#018571",     # green
        2000: "#7b3294",  # purple
        5000: "#2166ac",  # blue
    }

    # compute percentiles
    percentiles = {
        buff: computepercentiles(
            ens,
            percentiles=list(percentile_levels)
        )
        for buff, ens in ensembles.items()
    }

    buffers = sorted(ensembles.keys())
    panel_labels = iter(string.ascii_uppercase)

    # legend handles 
    buffer_handles = [
        Line2D(
            [0], [0],
            color=color_by_buffer[buff],
            label=f"+{buff/1000:.0f} km"
        )
        for buff in buffers
    ]

    observed_handle = Line2D(
        [0], [0],
        color=observed_color,
        label="Observed"
    )

    # OVERLAY MODE
    if mode == "overlay":

        fig, axes = plt.subplots(1, 3, figsize=figsize_overlay)

        #- plotting-
        for buff in buffers:
            c = color_by_buffer[buff]

            plotarearatio(
                ax=axes[0],
                ensemble=ensembles[buff],
                percentiles=percentiles[buff],
                observed=observed,
                synthetic_color=c,
                observed_color=observed_color,
            )

            plotprctinside(
                ax=axes[1],
                ensemble=ensembles[buff],
                synthetic_color=c,
            )

            plotdistanceoutside(
                ax=axes[2],
                ensemble=ensembles[buff],
                observed=observed,
                synthetic_color=c,
                observed_color=observed_color,
            )

        # lettering & grid
        for ax in axes:
            ax.text(
                0.01, 0.98,
                next(panel_labels),
                transform=ax.transAxes,
                ha="left", va="top"
            )
            ax.grid(True, alpha=0.3)

        # legends
        axes[0].legend(
            handles=buffer_handles + ([observed_handle] if observed is not None else []),
            frameon=False
        )

        axes[1].legend(
            handles=buffer_handles,
            frameon=False
        )

        axes[2].legend(
            handles=buffer_handles + ([observed_handle] if observed is not None else []),
            frameon=False
        )

        plt.tight_layout()
        plt.show()
        return fig

    # SEPARATE MODE
    elif mode == "separate":

        fig, axes = plt.subplots(
            len(buffers), 3,
            figsize=figsize_separate,
            sharex="col"
        )

        for row, buff in enumerate(buffers):
            c = color_by_buffer[buff]

            plotarearatio(
                ax=axes[row, 0],
                ensemble=ensembles[buff],
                percentiles=percentiles[buff],
                observed=observed,
                synthetic_color=c,
                observed_color=observed_color,
            )


            plotprctinside(
                ax=axes[row, 1],
                ensemble=ensembles[buff],
                synthetic_color=c,
            )

            plotdistanceoutside(
                ax=axes[row, 2],
                ensemble=ensembles[buff],
                observed=observed,
                synthetic_color=c,
                observed_color=observed_color,
            )

            for col in range(3):
                ax = axes[row, col]
                ax.text(
                    0.01, 0.98,
                    next(panel_labels),
                    transform=ax.transAxes,
                    ha="left", va="top"
                )
                ax.grid(True, alpha=0.3)

                # add legend titles indicating buffer distance
                buffer_label = f"+{buff/1000:.0f} km"

                # Area ratio panel (A)
                leg = axes[row, 0].get_legend()
                if leg is not None:
                    leg.set_title(buffer_label)

                # % outside panel (B)
                # intentionally no legend — do nothing

                # Distance outside panel (C)
                leg = axes[row, 2].get_legend()
                if leg is not None:
                    leg.set_title(buffer_label)

        plt.tight_layout()
        plt.show()
        return fig

    else:
        raise ValueError("mode must be 'overlay' or 'separate'")


def plotdistprevhist(ensemble: pd.DataFrame,
                     bins: int | str = 'auto',
                     figsize: tuple[float, float] = (7.5, 4.5),
                     savepath: str | None = None):
    """
    Histogram of distance to previous vent (metres) across all simulations and times.
    Y-axis in percent. Annotates mean, std, and N.
    """

    d = ensemble['distprev'].to_numpy(dtype=float)
    d = d[np.isfinite(d)]
    d = d[d > 0.0]  

    N = d.size
    fig, ax = plt.subplots(figsize=figsize)
    if N == 0:
        ax.text(0.5, 0.5, 'No distances to plot', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_axis_off()
        if savepath:
            fig.savefig(savepath, dpi=200, bbox_inches='tight')
        return fig

    mu = float(d.mean())
    sd = float(d.std(ddof=1)) if N > 1 else 0.0

    w = np.ones_like(d) * (100.0 / N)  
    ax.hist(d, bins=bins, weights=w, color='#4c78a8', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('Distance to previous vent (m)')
    ax.set_ylabel('Frequency [%]')
    ax.grid(True, axis='y', alpha=0.25)

    txt = f'N = {N:,}\nmean = {mu:,.1f} m\nstd = {sd:,.1f} m'
    bbox = dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='0.7', alpha=0.9)
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha='right', va='top', fontsize=10, bbox=bbox)

    if savepath:
        fig.savefig(savepath, dpi=200, bbox_inches='tight')
    return fig