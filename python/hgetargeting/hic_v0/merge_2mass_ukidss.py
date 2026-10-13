import numpy as np
from scipy.spatial import cKDTree
from astropy.table import vstack, MaskedColumn


def mag_to_flux(mag):
    return 10.0 ** (-0.4 * np.asarray(mag))


def flux_to_mag(flux):
    return -2.5 * np.log10(np.asarray(flux))


def radec_to_unitvec(ra_deg, dec_deg):
    ra = np.deg2rad(np.asarray(ra_deg))
    dec = np.deg2rad(np.asarray(dec_deg))
    cosd = np.cos(dec)
    return np.vstack((cosd * np.cos(ra), cosd * np.sin(ra), np.sin(dec))).T


def arcsec_to_chord(radius_arcsec):
    theta = np.deg2rad(radius_arcsec / 3600.0)
    return 2.0 * np.sin(theta / 2.0)


def chord_to_arcsec(chord):
    theta = 2.0 * np.arcsin(np.clip(np.asarray(chord) / 2.0, 0, 1))
    return np.rad2deg(theta) * 3600.0


def _make_string_column(table, name, length=80, default=""):
    table[name] = np.full(len(table), default, dtype=f"U{length}")




def _resolve_column(table, preferred=None, aliases=(), required=False, label="column"):
    """
    Return the first matching column name.

    Matching is tried in this order:
      1. preferred, if supplied
      2. aliases, case-sensitive
      3. aliases, case-insensitive

    If no match is found, return None unless required=True.
    """
    if preferred is not None and preferred in table.colnames:
        return preferred

    for name in aliases:
        if name in table.colnames:
            return name

    lower_map = {name.lower(): name for name in table.colnames}
    if preferred is not None and preferred.lower() in lower_map:
        return lower_map[preferred.lower()]

    for name in aliases:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    if required:
        raise ValueError(
            f"Could not find {label}. Tried preferred={preferred!r}, "
            f"aliases={list(aliases)!r}. Available columns are: {table.colnames}"
        )

    return None


def add_2mass_photometry_columns(table):
    """
    Add common-system 2MASS J/H/Ks columns while preserving native magnitudes.
    """
    n = len(table)

    table["jmag_native"] = np.asarray(table["jmag"], dtype=float)
    table["hmag_native"] = np.asarray(table["hmag"], dtype=float)
    table["kmag_native"] = np.asarray(table["kmag"], dtype=float)
    table["native"] = 'ukidss'
    
    j = np.asarray(table["jmag_native"], dtype=float)
    h = np.asarray(table["hmag_native"], dtype=float)
    k = np.asarray(table["kmag_native"], dtype=float)

    j_out = np.full(n, np.nan, dtype=float)
    h_out = np.full(n, np.nan, dtype=float)
    k_out = np.full(n, np.nan, dtype=float)

    good_jk = np.isfinite(j) & np.isfinite(k)
    good_hk = np.isfinite(h) & np.isfinite(k)

    # UKIDSS -> 2MASS color equations.
    # From Hewett+2006
    # J_2mass = J_ukidss + 0.02
    # H_2mass = H_ukidss - 0.01 - 0.063*(H-K)_ukidss
    # K_2mass = K_ukidss - 0.072*(H-K)_ukidss
    # Using actual data
    # jfixed = tab2['jmag'] + 0.01416253 + 0.01012938*(tab2['jmag']-tab2['kmag'])
    # hfixed = tab2['hmag'] - 0.04141157 - 0.07741838*(tab2['hmag']-tab2['kmag'])
    # kfixed = tab2['kmag'] - 0.02834592 + 0.01103323*(tab2['hmag']-tab2['kmag'])

    j_out[good_jk] = j[good_jk] + (j[good_jk] - k[good_jk])*0.0101 + 0.014
    h_out[good_hk] = h[good_hk] + (h[good_hk] - k[good_hk])*(-0.0774) - 0.0414
    k_out[good_hk] = k[good_hk] + (h[good_hk] - k[good_hk])*0.0110 - 0.0283
    
    # Fall back to native values when a color term cannot be computed.
    # This preserves usable H even if J or Ks is missing.
    bad = ~np.isfinite(j_out) & np.isfinite(j)
    j_out[bad] = j[bad]
    bad = ~np.isfinite(h_out) & np.isfinite(h)
    h_out[bad] = h[bad]
    bad = ~np.isfinite(k_out) & np.isfinite(k)
    k_out[bad] = k[bad]

    # 2mass names
    table["jmag_2mass"] = j_out
    table["hmag_2mass"] = h_out
    table["kmag_2mass"] = k_out

    return table

def merge(
    twomass,
    ukidss,
    use_common_photometry=True,
    common_h_col="hmag_2mass",
    h_bright=12.8,
    h_faint=13.2,
    h_match_min=12.5,
    h_match_max=13.5,
    search_radius_arcsec=1.0,
    good_match_radius_arcsec=0.5,
    max_dh_single=0.3,
    max_dh_blend=0.3,
    faint_blend_limit=3.0,
    dominant_frac=0.9,
    bright_blend_hmin=None,
    bright_blend_min_neighbors=2,
    bright_blend_max_dh=0.3,
    bright_blend_dominant_frac=0.8,
    prefer_ukidss_in_transition=True,
    verbose=True
):
    """
    Merge 2MASS and UKIDSS

    adds robust common-system UKIDSS photometry columns while preserving native
    2MASS/UKIDSS magnitudes.  It also appends nearest Gaia counterpart columns
    to the final merged catalog.  The Gaia match is performed to each output
    row's own position.

    Policy:
      * H_2MASS < h_bright: always keep 2MASS.  UKIDSS/Gaia only add blend flags.
      * h_bright <= H_2MASS <= h_faint: true replacement/deblend zone.
      * H_UKIDSS > h_faint: keep UKIDSS.

    Gaia is used only for source multiplicity/astrometric diagnostics, not for
    the IR photometric handoff.

    bright_blend_hmin:
      If None, all H_2MASS < h_bright sources are checked for bright-blend
      diagnostics.  If set, only bright sources with
      bright_blend_hmin <= H_2MASS < h_bright are checked.  This lets you avoid
      using UKIDSS diagnostics where UKIDSS is too saturated, e.g. set 12.0 or 12.3.
    """
    if h_match_min > h_bright:
        raise ValueError("h_match_min should normally be <= h_bright")
    if h_match_max < h_faint:
        raise ValueError("h_match_max should normally be >= h_faint")
    if h_bright > h_faint:
        raise ValueError("h_bright must be <= h_faint")

    twomass = twomass.copy(copy_data=False)
    ukidss = ukidss.copy(copy_data=False)

    # Add native and common-system photometric columns.
    # UKIDSS magnitudes are already on the UKIDSS system.  2MASS is transformed.
    #add_ukidss_photometry_columns(twomass, "2MASS", j_col=j_col, h_col=h_col, ks_col=ks_col)
    add_2mass_photometry_columns(ukidss)

    twomass['jmag_2mass'] = np.asarray(twomass['jmag'], dtype=float)
    twomass['hmag_2mass'] = np.asarray(twomass['hmag'], dtype=float)
    twomass['kmag_2mass'] = np.asarray(twomass['kmag'], dtype=float)
    twomass['native'] = '2mass'
    
    # Use common-system H for the merge thresholds by default.
    h_merge_col = common_h_col if use_common_photometry else h_col
    if h_merge_col not in twomass.colnames or h_merge_col not in ukidss.colnames:
        raise ValueError(f"Requested H merge column {h_merge_col!r} not present in both catalogs")

    n2, nv = len(twomass), len(ukidss)

    # 2MASS-side diagnostics.  These are propagated to output rows.
    twomass["n_ukidss_neighbors"] = np.zeros(n2, dtype=np.int32)
    twomass["ukidss_blend_h"] = np.full(n2, np.nan, dtype=float)
    twomass["ukidss_brightest_frac"] = np.full(n2, np.nan, dtype=float)
    twomass["ukidss_nearest_sep_arcsec"] = np.full(n2, np.nan, dtype=float)
    twomass["blend_score"] = np.zeros(n2, dtype=np.int16)

    tm_xyz = radec_to_unitvec(twomass["ra"], twomass["dec"])
    ukidss_xyz = radec_to_unitvec(ukidss["ra"], ukidss["dec"])
    ukidss_tree = cKDTree(ukidss_xyz)
    search_chord = arcsec_to_chord(search_radius_arcsec)

    used_2mass = np.zeros(n2, dtype=bool)
    used_ukidss = np.zeros(nv, dtype=bool)

    out_cat, out_ind, out_flag = [], [], []
    out_m2, out_mv = [], []
    out_nukidss = []
    out_hblend, out_bfrac = [], []
    out_vsep, out_gsep, out_bscore = [], [], []

    def add_2mass(i, flag, matched_ukidss=-1):
        if used_2mass[i]:
            return
        out_cat.append("2MASS")
        out_ind.append(i)
        out_flag.append(flag)
        out_m2.append(-1)
        out_mv.append(matched_ukidss)
        out_nukidss.append(twomass["n_ukidss_neighbors"][i])
        out_hblend.append(twomass["ukidss_blend_h"][i])
        out_bfrac.append(twomass["ukidss_brightest_frac"][i])
        out_vsep.append(twomass["ukidss_nearest_sep_arcsec"][i])
        out_bscore.append(twomass["blend_score"][i])
        used_2mass[i] = True

    def add_ukidss(j, flag, matched_2mass=-1):
        if used_ukidss[j]:
            return
        out_cat.append("UKIDSS")
        out_ind.append(j)
        out_flag.append(flag)
        out_m2.append(matched_2mass)
        out_mv.append(-1)
        if matched_2mass >= 0:
            i = matched_2mass
            out_nukidss.append(twomass["n_ukidss_neighbors"][i])
            out_hblend.append(twomass["ukidss_blend_h"][i])
            out_bfrac.append(twomass["ukidss_brightest_frac"][i])
            out_vsep.append(twomass["ukidss_nearest_sep_arcsec"][i])
            out_bscore.append(twomass["blend_score"][i])
        else:
            out_nukidss.append(0)
            out_hblend.append(np.nan)
            out_bfrac.append(np.nan)
            out_vsep.append(np.nan)
            out_bscore.append(0)
        used_ukidss[j] = True

    def get_neighbors_for_2mass(i):
        h2 = twomass[h_merge_col][i]
        nbrs = np.asarray(ukidss_tree.query_ball_point(tm_xyz[i], r=search_chord), dtype=int)
        if len(nbrs) > 0:
            nbrs = nbrs[ukidss[h_merge_col][nbrs] <= h2 + faint_blend_limit]
            sep_arcsec = chord_to_arcsec(np.linalg.norm(ukidss_xyz[nbrs] - tm_xyz[i], axis=1))
        else:
            sep_arcsec = np.array([], dtype=float)

        twomass["n_ukidss_neighbors"][i] = len(nbrs)
        if len(sep_arcsec) > 0:
            twomass["ukidss_nearest_sep_arcsec"][i] = np.min(sep_arcsec)

        if len(nbrs) > 0:
            fluxes = mag_to_flux(ukidss[h_merge_col][nbrs])
            htot = flux_to_mag(np.sum(fluxes))
            bfrac = np.max(fluxes) / np.sum(fluxes)
            twomass["ukidss_blend_h"][i] = htot
            twomass["ukidss_brightest_frac"][i] = bfrac
        else:
            htot = np.nan
            bfrac = np.nan

        gnbrs = np.array([], dtype=int)
        gsep_arcsec = np.array([], dtype=float)
            
        score = 0
        if len(nbrs) >= bright_blend_min_neighbors:
            score += 1
        if len(gnbrs) >= bright_blend_min_neighbors:
            score += 1
        if len(nbrs) >= 2 and np.isfinite(htot) and abs(h2 - htot) <= bright_blend_max_dh:
            score += 1
        if len(nbrs) >= 2 and np.isfinite(bfrac) and bfrac < bright_blend_dominant_frac:
            score += 1
        twomass["blend_score"][i] = score
        return nbrs, sep_arcsec, gnbrs, gsep_arcsec

    # 1. Bright 2MASS diagnostics only; never replace with UKIDSS here.
    if bright_blend_hmin is None:
        bright_check = np.where(twomass[h_merge_col] < h_bright)[0]
    else:
        bright_check = np.where((twomass[h_merge_col] >= bright_blend_hmin) & (twomass[h_merge_col] < h_bright))[0]

    if verbose:
        print(f"Checking {len(bright_check)} bright 2MASS sources for blend diagnostics")
    for kk, i in enumerate(bright_check):
        if verbose and kk % 5000 == 0:
            print(f"bright diagnostic {kk} / {len(bright_check)}")
        get_neighbors_for_2mass(i)
        if twomass["blend_score"][i] >= 3:
            flag = "bright_2MASS_probable_blend_keep_2MASS"
        elif twomass["blend_score"][i] >= 2:
            flag = "bright_2MASS_possible_blend_keep_2MASS"
        else:
            flag = "bright_2MASS"
        add_2mass(i, flag)

    # 2. Transition diagnostic/replacement zone.
    tm_check = np.where((twomass[h_merge_col] >= h_match_min) & (twomass[h_merge_col] <= h_match_max) & (~used_2mass))[0]
    if verbose:
        print(f"Checking {len(tm_check)} 2MASS sources in transition diagnostic zone")
    for kk, i in enumerate(tm_check):
        if verbose and kk % 2000 == 0:
            print(f"transition diagnostic {kk} / {len(tm_check)}")
        h2 = twomass[h_merge_col][i]
        in_replace_zone = (h2 >= h_bright) and (h2 <= h_faint)
        nbrs, sep_arcsec, _, _ = get_neighbors_for_2mass(i)

        if len(nbrs) == 0:
            add_2mass(i, "2MASS_checked_no_ukidss")
            continue

        fluxes = mag_to_flux(ukidss[h_merge_col][nbrs])
        htot = flux_to_mag(np.sum(fluxes))
        brightest_local = np.argmax(fluxes)
        brightest_ukidss = nbrs[brightest_local]
        brightest_frac = fluxes[brightest_local] / np.sum(fluxes)
        dh_blend = h2 - htot

        if len(nbrs) == 1:
            j = nbrs[0]
            sep = sep_arcsec[0]
            dh = h2 - ukidss[h_merge_col][j]
            is_clean = (sep <= good_match_radius_arcsec) and (abs(dh) <= max_dh_single)

            if is_clean and in_replace_zone and prefer_ukidss_in_transition:
                add_ukidss(j, "clean_2MASS_UKIDSS_match", matched_2mass=i)
                used_2mass[i] = True
            elif is_clean and in_replace_zone and not prefer_ukidss_in_transition:
                add_2mass(i, "clean_2MASS_UKIDSS_match_keep_2MASS", matched_ukidss=j)
            elif is_clean and h2 < h_bright:
                add_2mass(i, "bright_diagnostic_clean_ukidss_keep_2MASS", matched_ukidss=j)
            elif is_clean and h2 > h_faint:
                add_ukidss(j, "faint_diagnostic_clean_ukidss", matched_2mass=i)
                used_2mass[i] = True
            else:
                if h2 < h_bright:
                    add_2mass(i, "bright_diagnostic_uncertain_single_ukidss")
                elif h2 > h_faint:
                    add_ukidss(j, "faint_diagnostic_uncertain_single_ukidss", matched_2mass=i)
                    used_2mass[i] = True
                else:
                    add_2mass(i, "transition_uncertain_single_ukidss", matched_ukidss=j)
                    add_ukidss(j, "transition_uncertain_single_2MASS", matched_2mass=i)
        else:
            if brightest_frac >= dominant_frac:
                j = brightest_ukidss
                sep = sep_arcsec[brightest_local]
                dh = h2 - ukidss[h_merge_col][j]
                is_clean_dom = (sep <= good_match_radius_arcsec) and (abs(dh) <= max_dh_single)

                if is_clean_dom and in_replace_zone and prefer_ukidss_in_transition:
                    add_ukidss(j, "dominant_ukidss_match", matched_2mass=i)
                    used_2mass[i] = True
                elif is_clean_dom and in_replace_zone and not prefer_ukidss_in_transition:
                    add_2mass(i, "dominant_ukidss_match_keep_2MASS", matched_ukidss=j)
                elif is_clean_dom and h2 < h_bright:
                    add_2mass(i, "bright_diagnostic_dominant_ukidss_keep_2MASS", matched_ukidss=j)
                elif is_clean_dom and h2 > h_faint:
                    add_ukidss(j, "faint_diagnostic_dominant_ukidss", matched_2mass=i)
                    used_2mass[i] = True
                else:
                    if h2 < h_bright:
                        add_2mass(i, "bright_diagnostic_uncertain_dominant_ukidss")
                    elif h2 > h_faint:
                        add_ukidss(j, "faint_diagnostic_uncertain_dominant_ukidss", matched_2mass=i)
                        used_2mass[i] = True
                    else:
                        add_2mass(i, "transition_uncertain_dominant_ukidss", matched_ukidss=j)
                        add_ukidss(j, "transition_uncertain_dominant_2MASS", matched_2mass=i)
            elif abs(dh_blend) <= max_dh_blend:
                if in_replace_zone:
                    for j in nbrs:
                        add_ukidss(j, "UKIDSS_component_of_2MASS_blend", matched_2mass=i)
                    used_2mass[i] = True
                elif h2 < h_bright:
                    flag = "bright_2MASS_probable_blend_keep_2MASS" if twomass["blend_score"][i] >= 3 else "bright_2MASS_possible_blend_keep_2MASS"
                    add_2mass(i, flag)
                else:
                    for j in nbrs:
                        add_ukidss(j, "faint_diagnostic_UKIDSS_components_of_2MASS_blend", matched_2mass=i)
                    used_2mass[i] = True
            else:
                if h2 < h_bright:
                    flag = "bright_2MASS_possible_blend_keep_2MASS" if twomass["blend_score"][i] >= 2 else "bright_diagnostic_blend_flux_mismatch_keep_2MASS"
                    add_2mass(i, flag)
                elif h2 > h_faint:
                    for j in nbrs:
                        add_ukidss(j, "faint_diagnostic_uncertain_blend_flux_mismatch", matched_2mass=i)
                    used_2mass[i] = True
                else:
                    add_2mass(i, "transition_uncertain_blend_flux_mismatch")
                    for j in nbrs:
                        add_ukidss(j, "transition_UKIDSS_near_uncertain_2MASS_blend", matched_2mass=i)

    # 3. Final additions.
    bright_2mass = np.where((twomass[h_merge_col] < h_bright) & (~used_2mass))[0]
    for i in bright_2mass:
        add_2mass(i, "bright_2MASS")

    transition_2mass = np.where((twomass[h_merge_col] >= h_bright) & (twomass[h_merge_col] <= h_faint) & (~used_2mass))[0]
    for i in transition_2mass:
        add_2mass(i, "transition_2MASS_no_UKIDSS")

    faint_ukidss = np.where((ukidss[h_merge_col] > h_faint) & (~used_ukidss))[0]
    for j in faint_ukidss:
        add_ukidss(j, "faint_UKIDSS")

    transition_ukidss = np.where((ukidss[h_merge_col] >= h_bright) & (ukidss[h_merge_col] <= h_faint) & (~used_ukidss))[0]
    for j in transition_ukidss:
        add_ukidss(j, "transition_UKIDSS_no_2MASS")

    if verbose:
        print(f"Using H column for merge thresholds: {h_merge_col}")
        print("Final additions:")
        print(f"  bright_2MASS:        {len(bright_2mass)}")
        print(f"  transition_2MASS:    {len(transition_2mass)}")
        print(f"  faint_UKIDSS:           {len(faint_ukidss)}")
        print(f"  transition_UKIDSS:      {len(transition_ukidss)}")


    # 4. Build output table.
    out_cat = np.asarray(out_cat)
    out_ind = np.asarray(out_ind, dtype=int)
    out_flag = np.asarray(out_flag, dtype="U80")
    out_m2 = np.asarray(out_m2, dtype=int)
    out_mv = np.asarray(out_mv, dtype=int)
    out_nukidss = np.asarray(out_nukidss, dtype=np.int32)
    out_hblend = np.asarray(out_hblend, dtype=float)
    out_bfrac = np.asarray(out_bfrac, dtype=float)
    out_vsep = np.asarray(out_vsep, dtype=float)
    out_bscore = np.asarray(out_bscore, dtype=np.int16)

    tables = []
    tm_mask = out_cat == "2MASS"
    uk_mask = out_cat == "UKIDSS"

    if np.any(tm_mask):
        tm_out = twomass[out_ind[tm_mask]].copy()
        _make_string_column(tm_out, "catalog", length=16, default="2MASS")
        _make_string_column(tm_out, "merge_flag", length=80)
        tm_out["merge_flag"] = out_flag[tm_mask]
        tm_out["matched_2mass_index"] = out_m2[tm_mask]
        tm_out["matched_ukidss_index"] = out_mv[tm_mask]
        tm_out["n_ukidss_neighbors"] = out_nukidss[tm_mask]
        tm_out["ukidss_blend_h"] = out_hblend[tm_mask]
        tm_out["ukidss_brightest_frac"] = out_bfrac[tm_mask]
        tm_out["ukidss_nearest_sep_arcsec"] = out_vsep[tm_mask]
        tm_out["blend_score"] = out_bscore[tm_mask]
        tables.append(tm_out)

    if np.any(uk_mask):
        uk_out = ukidss[out_ind[uk_mask]].copy()
        _make_string_column(uk_out, "catalog", length=16, default="UKIDSS")
        _make_string_column(uk_out, "merge_flag", length=80)
        uk_out["merge_flag"] = out_flag[uk_mask]
        uk_out["matched_2mass_index"] = out_m2[uk_mask]
        uk_out["matched_ukidss_index"] = out_mv[uk_mask]
        uk_out["n_ukidss_neighbors"] = out_nukidss[uk_mask]
        uk_out["ukidss_blend_h"] = out_hblend[uk_mask]
        uk_out["ukidss_brightest_frac"] = out_bfrac[uk_mask]
        uk_out["ukidss_nearest_sep_arcsec"] = out_vsep[uk_mask]
        uk_out["blend_score"] = out_bscore[uk_mask]
        tables.append(uk_out)

    if len(tables) == 0:
        return twomass[:0].copy()
    return vstack(tables, metadata_conflicts="silent")
