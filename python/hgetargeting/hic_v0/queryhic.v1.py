import os
import numpy as np
import healpy as hp
from astropy.table import Table,vstack,MaskedColumn,Column
from astropy.coordinates import SkyCoord

_CATALOGS = {"virac2", "vvvx", "ukidss"}

# Shared HIC fields supplied in the three examples. Edit this tuple if the
# shared HIC schema changes. Catalog-specific fields are defined below.
_BASE_COLUMNS = (
    "ra", "dec", "designation", "jmag", "jerr", "hmag", "herr",
    "kmag", "kerr", "ph_qual", "rd_flg", "jmag_2mass",
    "hmag_2mass", "kmag_2mass", "native", "nircatalog",
    "jmag_native", "hmag_native", "kmag_native",
    "glimpse_designation", "glimpse_mag36", "glimpse_mag36err",
    "glimpse_mag45", "glimpse_mag45err", "unwise_objid",
    "unwise_primary_status", "unwise_flux_w1", "unwise_dflux_w1",
    "unwise_flux_w2", "unwise_dflux_w2", "unwise_fracflux_w1",
    "unwise_fracflux_w2", "unwise_qf_w1", "unwise_qf_w2",
    "unwise_rchi2_w1", "unwise_rchi2_w2", "unwise_flags_unwise_w2",
    "unwise_flags_info_w2", "unwise_w1mag", "unwise_w1err",
    "unwise_w2mag", "unwise_w2err", "mircatalog", "hpix",
)

_COMMON = {
    "blend_score": "nir_blend_score",
    "catalog": "nir_catalog",
    "merge_flag": "nir_merge_flag",
    "matched_2mass_index": "nir_matched_2mass_index",
}

_CATALOG_COLUMNS = {
    "ukidss": {
        "n_ukidss_neighbors": "nir_n_neighbors",
        "ukidss_blend_h": "nir_blend_h",
        "ukidss_brightest_frac": "nir_brightest_frac",
        "ukidss_nearest_sep_arcsec": "nir_nearest_sep_arcsec",
        "matched_ukidss_index": "nir_matched_deep_index",
        "sourceid": "nir_sourceid",
        **{name: "ukidss_" + name for name in (
            "jpperrbits", "hpperrbits", "kpperrbits", "mergedclass",
            "mergedclassstat", "pstar", "jclass", "hclass", "kclass",
            "jell", "hell", "kell",
        )},
    },
    "virac2": {
        "n_vvv_neighbors": "nir_n_neighbors",
        "vvv_blend_h": "nir_blend_h",
        "vvv_brightest_frac": "nir_brightest_frac",
        "vvv_nearest_sep_arcsec": "nir_nearest_sep_arcsec",
        "matched_vvv_index": "nir_matched_deep_index",
        "sourceid": "nir_sourceid",
        "chisq": "virac2_chisq",
        "uwe": "virac2_uwe",
    },
    "vvvx": {
        "n_vvv_neighbors": "nir_n_neighbors",
        "vvv_blend_h": "nir_blend_h",
        "vvv_brightest_frac": "nir_brightest_frac",
        "vvv_nearest_sep_arcsec": "nir_nearest_sep_arcsec",
        "matched_vvv_index": "nir_matched_deep_index",
        "uniqueid": "nir_sourceid",
        **{name: "vvvx_" + name for name in (
            "cueventid", "framesetid", "cx", "cy", "cz",
            "htmid", "l", "b", "priorsec", "jmhpnt", "jmhpnterr",
            "hmkspnt", "hmkspnterr", "jmkspnt", "jmkspnterr",
        )},
    },
}

# Columns that can differ among the three deep-catalog inputs. Every output
# receives the complete set. A missing value is masked, never set to zero.
_CANONICAL_COLUMNS = set(_COMMON.values())
for _mapping in _CATALOG_COLUMNS.values():
    _CANONICAL_COLUMNS.update(_mapping.values())

# Used only when a column is absent. Existing columns retain their dtype.
_STRING_COLUMNS = {"nir_catalog"}
_INTEGER_COLUMNS = {"nir_n_neighbors", "nir_matched_2mass_index",
    "nir_matched_deep_index", "nir_sourceid",
                    "vvvx_cueventid", "vvvx_framesetid",
                    "ukidss_jpperrbits",
                    "ukidss_hpperrbits", "ukidss_kpperrbits",
                    "ukidss_mergedclass", "ukidss_jclass", "ukidss_hclass",
                    "ukidss_kclass", "vvvx_priorsec", "vvvx_htmid"}

STANDARD_DTYPE = [('hic_designation', '<U33'),('ra', '<f8'), ('dec', '<f8'), ('hpix', '<i8'), ('designation','S27'),
                  ('jmag_native', '<f8'), ('jerr', '<f8'),('hmag_native','<f8'), ('herr', '<f8'),
                  ('kmag_native', '<f8'), ('kerr', '<f8'),('ph_qual', 'S13'), ('rd_flg', 'S13'),
                  ('jmag', '<f8'), ('hmag','<f8'),('kmag', '<f8'), ('native', 'S6'), ('nircatalog', 'S6'),
                  ('nir_blend_h','<f8'),('nir_blend_score', '<i2'), ('nir_brightest_frac', '<f8'),
                  ('nir_catalog', 'S16'), ('nir_matched_2mass_index', '<i8'),
                  ('nir_matched_deep_index', '<i8'), ('nir_merge_flag', 'S80'),
                  ('nir_n_neighbors', '<i4'), ('nir_nearest_sep_arcsec', '<f8'),
                  ('nir_sourceid', '<i8'), ('ukidss_hclass', '<i8'), ('ukidss_hell','<f8'),
                  ('ukidss_hpperrbits', '<i8'), ('ukidss_jclass', '<i8'),
                  ('ukidss_jell', '<f8'), ('ukidss_jpperrbits', '<i8'),
                  ('ukidss_kclass', '<i8'), ('ukidss_kell', '<f8'),
                  ('ukidss_kpperrbits', '<i8'), ('ukidss_mergedclass', '<i8'),
                  ('ukidss_mergedclassstat', '<f8'), ('ukidss_pstar', '<f8'),
                  ('virac2_chisq', '<f8'), ('virac2_uwe', '<f8'), 
                  ('vvvx_cueventid', '<i8'), ('vvvx_framesetid', '<i8'), ('vvvx_hmkspnt','<f8'),
                  ('vvvx_hmkspnterr', '<f8'), ('vvvx_htmid', '<i8'),
                  ('vvvx_jmhpnt', '<f8'), ('vvvx_jmhpnterr', '<f8'), ('vvvx_jmkspnt','<f8'),
                  ('vvvx_jmkspnterr', '<f8'), ('vvvx_priorsec', '<i8'),
                  ('glimpse_designation', 'S43'), ('glimpse_mag36', '<f4'),
                  ('glimpse_mag36err', '<f4'), ('glimpse_mag45', '<f4'),
                  ('glimpse_mag45err', '<f4'), ('unwise_objid', 'S26'),
                  ('unwise_primary_status', '<i8'), ('unwise_flux_w1', '<f4'),
                  ('unwise_dflux_w1', '<f4'), ('unwise_flux_w2', '<f4'),
                  ('unwise_dflux_w2', '<f4'), ('unwise_fracflux_w1', '<f4'),
                  ('unwise_fracflux_w2', '<f4'), ('unwise_qf_w1', '<f4'),
                  ('unwise_qf_w2', '<f4'), ('unwise_rchi2_w1', '<f4'),
                  ('unwise_rchi2_w2', '<f4'), ('unwise_flags_unwise_w2', '<f4'),
                  ('unwise_flags_info_w2', '<f4'), ('unwise_w1mag', '<f4'),
                  ('unwise_w1err', '<f4'), ('unwise_w2mag', '<f4'), ('unwise_w2err','<f4'),('mircatalog', 'S10')]

def standardize_hic_nir(table: Table, deep_catalog: str) -> Table:
    """Return a copy with common NIR names and catalog-prefixed native fields.

    Missing catalog-specific fields are masked. Raw aliases are consumed even
    if a prior processing step has already added the standardized field.
    Only the fixed base and NIR quality schema is returned.
    """
    deep_catalog = deep_catalog.lower()
    if deep_catalog not in _CATALOGS:
        raise ValueError(f"deep_catalog must be one of {sorted(_CATALOGS)}")

    out = table
    mapping = dict(_COMMON)
    mapping.update(_CATALOG_COLUMNS[deep_catalog])
    missing_base = sorted(set(_BASE_COLUMNS) - set(out.colnames))
    if missing_base:
        raise ValueError("Missing shared HIC columns: " + ", ".join(missing_base))
    for old, new in mapping.items():
        if old not in out.colnames:
            continue
        if new in out.colnames:
            # Raw value wins if both old and new appear in an input table.
            out.replace_column(new, out[old].copy())
            out.remove_column(old)
        else:
            out.rename_column(old, new)

    for name in sorted(_CANONICAL_COLUMNS - set(out.colnames)):
        dtype = "U64" if name in _STRING_COLUMNS else (
            "i8" if name in _INTEGER_COLUMNS else "f8")
        out.add_column(MaskedColumn(np.zeros(len(out), dtype=dtype),
                                    name=name, mask=True))
    # Remove the redundant field from tables previously standardized by v4.
    if "vvvx_uniqueid" in out.colnames:
        del out["vvvx_uniqueid"]
    out = out[list(_BASE_COLUMNS) + sorted(_CANONICAL_COLUMNS)]
    # Rename jmag/hmag/kmag to jmag_native/hmag_native/kmag_native
    # and Xmag_2mass to Xmag
    del out[['jmag_native','hmag_native','kmag_native']]  # only the native deep nir data
    out['jmag'].name = 'jmag_native'
    out['hmag'].name = 'hmag_native'
    out['kmag'].name = 'kmag_native'
    out['jmag_2mass'].name = 'jmag'
    out['hmag_2mass'].name = 'hmag'
    out['kmag_2mass'].name = 'kmag'
    
    return out

def _value(value):
    """Convert an unmasked table scalar to a nonempty string."""
    if np.ma.is_masked(value) or value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).strip()
    return value if value and value.lower() not in ("nan", "none") else None

def _check_unique(names):
    seen = set()
    for i, name in enumerate(names):
        if name in seen:
            raise ValueError("Duplicate HIC designation at row {}: {}".format(i, name))
        seen.add(name)

def add_hic_designation(table, deep_catalog, output_col="hic_designation"):
    """Return an Astropy Table copy with a namespaced source designation.

    ``native`` is '2mass', 'vvv', or 'ukidss'. The deep catalog identifier is
    always in ``nir_sourceid``. For 'vvv', deep_catalog specifies whether the
    table is VIRAC2 or VVVX. No per-row Python loop or duplicate scan is done.
    """
    deep_catalog = deep_catalog.lower()
    if deep_catalog not in ("virac2", "vvvx", "ukidss"):
        raise ValueError("Unknown deep_catalog: " + deep_catalog)
    required = {"native", "designation", "nir_sourceid"}
    missing = required - set(table.colnames)
    if missing:
        raise ValueError("Missing columns: " + ", ".join(sorted(missing)))

    native = np.char.lower(np.char.strip(np.asarray(table["native"]).astype(str)))
    is_2mass = native == "2mass"
    deep_label = "ukidss" if deep_catalog == "ukidss" else "vvv"
    is_deep = native == deep_label
    if np.any(~(is_2mass | is_deep)):
        bad = np.flatnonzero(~(is_2mass | is_deep))[0]
        raise ValueError("Unknown native value at row {}: {!r}".format(
            bad, native[bad]))

    mass_id = np.char.strip(np.asarray(table["designation"]).astype(str))
    deep_id = np.char.strip(np.asarray(table["nir_sourceid"]).astype(str))
    mass_missing = np.ma.getmaskarray(table["designation"]) | (mass_id == "")
    deep_missing = np.ma.getmaskarray(table["nir_sourceid"]) | (deep_id == "")
    bad_id = (is_2mass & mass_missing) | (is_deep & deep_missing)
    if np.any(bad_id):
        raise ValueError("Missing source ID at row {}".format(np.flatnonzero(bad_id)[0]))

    # The prefix makes an ID unambiguous across all three input catalogs.
    prefix = deep_catalog.upper() + ":"
    designation = np.where(is_2mass,
                           np.char.add("2MASS:", mass_id),
                           np.char.add(prefix, deep_id))
    table[output_col] = designation
    return table

def cone_search(ra, dec, ra0, dec0, radius_deg):
    """Return indices of sources within radius_deg. Angles are in degrees."""
    ra = np.deg2rad(ra)
    dec = np.deg2rad(dec)
    ra0 = np.deg2rad(ra0)
    dec0 = np.deg2rad(dec0)
    radius = np.deg2rad(radius_deg)

    a = (np.sin((dec - dec0) / 2)**2
         + np.cos(dec) * np.cos(dec0)
         * np.sin((ra - ra0) / 2)**2)

    return np.flatnonzero(a <= np.sin(radius / 2)**2)

def query(lon,lat,radius,galactic=False):
    """ Query the HIC table. """
    if galactic:
        coo = SkyCoord(lon,lat,unit='deg',frame='galactic')
        ra = coo.icrs.ra.degree
        dec = coo.icrs.dec.degree
    else:
        ra = lon
        dec = lat
    
    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/hic/ring256/'
    pix = hp.ang2pix(256,ra,dec,lonlat=True)
    vec = hp.pix2vec(256,pix)
    querypix = hp.query_disc(256,vec,np.deg2rad(radius)*1.05,inclusive=True)
    print('Loading',len(querypix),'pixels')
    
    hic = []
    for i in range(len(querypix)):
        filename = basedir+'/hic_{:d}.fits'.format(querypix[i])
        if os.path.exists(filename):
            tab = Table.read(filename)
            tab['hpix'] = querypix[i]
            # Standardize the columns
            
            # UKIDSS
            if 'hell' in tab.colnames:
                deepnircatalog = 'ukidss'
            # VIRAC2
            elif 'uwe' in tab.colnames:
                deepnircatalog = 'virac2'            
            # vvvx
            elif 'jmhpnt' in tab.colnames:
                deepnircatalog = 'vvvx'
            else:
                print('do not recognize this format')
                
            # Standardize the columns
            tab = standardize_hic_nir(tab, deepnircatalog)
            # Add HIC ID
            tab = add_hic_designation(tab, deepnircatalog)
        else:
            print(filename,'not found')
            continue
        hic.append(tab)
    hic = vstack(hic)

    # Only take the sources within the desired radius
    idx = cone_search(hic["ra"], hic["dec"], ra, dec, radius)
    hic = hic[idx]
    
    return hic
