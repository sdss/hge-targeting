import os
import numpy as np
import healpy as hp
from astropy.table import Table,vstack,MaskedColumn,Column
from astropy.coordinates import SkyCoord

STANDARD_DTYPE = np.dtype([
    ('hic_designation', 'U33'), ('ra', 'f8'), ('dec', 'f8'), ('hpix', 'i8'),
    ('designation', 'S27'), ('jmag_native', 'f4'), ('jerr', 'f4'),
    ('hmag_native', 'f4'), ('herr', 'f4'), ('kmag_native', 'f4'), ('kerr', 'f4'),
    ('ph_qual', 'S13'), ('rd_flg', 'S13'), ('jmag', 'f4'), ('hmag', 'f4'),
    ('kmag', 'f4'), ('native', 'S6'), ('nircatalog', 'S6'),
    ('nir_blend_h', 'f4'), ('nir_blend_score', 'i2'),
    ('nir_brightest_frac', 'f4'), ('nir_catalog', 'S16'),
    ('nir_matched_2mass_index', 'i8'), ('nir_matched_deep_index', 'i8'),
    ('nir_merge_flag', 'S80'), ('nir_n_neighbors', 'i4'),
    ('nir_nearest_sep_arcsec', 'f4'), ('nir_sourceid', 'i8'),
    ('ukidss_hclass', 'i8'), ('ukidss_hell', 'f8'),
    ('ukidss_hpperrbits', 'i8'), ('ukidss_jclass', 'i8'),
    ('ukidss_jell', 'f8'), ('ukidss_jpperrbits', 'i8'),
    ('ukidss_kclass', 'i8'), ('ukidss_kell', 'f8'),
    ('ukidss_kpperrbits', 'i8'), ('ukidss_mergedclass', 'i8'),
    ('ukidss_mergedclassstat', 'f8'), ('ukidss_pstar', 'f8'),
    ('virac2_chisq', 'f8'), ('virac2_uwe', 'f8'),
    ('vvvx_cueventid', 'i8'), ('vvvx_framesetid', 'i8'),
    ('vvvx_hmkspnt', 'f8'), ('vvvx_hmkspnterr', 'f8'),
    ('vvvx_htmid', 'i8'), ('vvvx_jmhpnt', 'f8'),
    ('vvvx_jmhpnterr', 'f8'), ('vvvx_jmkspnt', 'f8'),
    ('vvvx_jmkspnterr', 'f8'), ('vvvx_priorsec', 'i8'),
    ('glimpse_designation', 'S43'), ('glimpse_mag36', 'f4'),
    ('glimpse_mag36err', 'f4'), ('glimpse_mag45', 'f4'),
    ('glimpse_mag45err', 'f4'), ('unwise_objid', 'S26'),
    ('unwise_primary_status', 'i8'), ('unwise_flux_w1', 'f4'),
    ('unwise_dflux_w1', 'f4'), ('unwise_flux_w2', 'f4'),
    ('unwise_dflux_w2', 'f4'), ('unwise_fracflux_w1', 'f4'),
    ('unwise_fracflux_w2', 'f4'), ('unwise_qf_w1', 'f4'),
    ('unwise_qf_w2', 'f4'), ('unwise_rchi2_w1', 'f4'),
    ('unwise_rchi2_w2', 'f4'), ('unwise_flags_unwise_w2', 'f4'),
    ('unwise_flags_info_w2', 'f4'), ('unwise_w1mag', 'f4'),
    ('unwise_w1err', 'f4'), ('unwise_w2mag', 'f4'),
    ('unwise_w2err', 'f4'), ('mircatalog', 'S10'),
])

_SHARED_ALIASES = {
    'nir_blend_score': ('blend_score',),
    'nir_catalog': ('catalog',),
    'nir_merge_flag': ('merge_flag',),
    'nir_matched_2mass_index': ('matched_2mass_index',),
}
_DEEP_ALIASES = {
    'ukidss': {
        'nir_n_neighbors': 'n_ukidss_neighbors',
        'nir_blend_h': 'ukidss_blend_h',
        'nir_brightest_frac': 'ukidss_brightest_frac',
        'nir_nearest_sep_arcsec': 'ukidss_nearest_sep_arcsec',
        'nir_matched_deep_index': 'matched_ukidss_index',
        'nir_sourceid': 'sourceid',
    },
    'virac2': {
        'nir_n_neighbors': 'n_vvv_neighbors',
        'nir_blend_h': 'vvv_blend_h',
        'nir_brightest_frac': 'vvv_brightest_frac',
        'nir_nearest_sep_arcsec': 'vvv_nearest_sep_arcsec',
        'nir_matched_deep_index': 'matched_vvv_index',
        'nir_sourceid': 'sourceid',
        'virac2_chisq': 'chisq',
        'virac2_uwe': 'uwe',
    },
    'vvvx': {
        'nir_n_neighbors': 'n_vvv_neighbors',
        'nir_blend_h': 'vvv_blend_h',
        'nir_brightest_frac': 'vvv_brightest_frac',
        'nir_nearest_sep_arcsec': 'vvv_nearest_sep_arcsec',
        'nir_matched_deep_index': 'matched_vvv_index',
        'nir_sourceid': 'uniqueid',
        **{ 'vvvx_' + x: x for x in (
            'cueventid', 'framesetid', 'hmkspnt', 'hmkspnterr', 'htmid',
            'jmhpnt', 'jmhpnterr', 'jmkspnt', 'jmkspnterr', 'priorsec')},
    },
}
_UKIDSS_NATIVE = ('hpperrbits', 'jpperrbits', 'kpperrbits',
                  'hclass', 'jclass', 'kclass', 'hell', 'jell', 'kell',
                  'mergedclass', 'mergedclassstat', 'pstar')
_REQUIRED = ('ra', 'dec', 'designation', 'native')


def _as_text(values):
    """Decode fixed-width byte strings without visiting Astropy Row objects."""
    return np.char.strip(np.asarray(values).astype(str))


def build_hic_table(table, deep_catalog):
    """Return a fixed-dtype structured array with bulk column assignments.

    Missing floats are NaN; missing integers are -1; missing byte strings are
    empty. Source masks are respected. ID errors are raised for rows that need
    an absent identifier. Input table is never modified.
    """
    deep_catalog = deep_catalog.lower()
    if deep_catalog not in _DEEP_ALIASES:
        raise ValueError('deep_catalog must be virac2, vvvx, or ukidss')
    columns = set(table.colnames if hasattr(table, 'colnames') else table.dtype.names)
    absent = set(_REQUIRED) - columns
    if absent:
        raise ValueError('Missing required columns: ' + ', '.join(sorted(absent)))

    out = np.empty(len(table), dtype=STANDARD_DTYPE)
    for name in STANDARD_DTYPE.names:
        kind = STANDARD_DTYPE[name].kind
        out[name].fill(np.nan if kind == 'f' else (-1 if kind in 'iu' else b''))
    out['hic_designation'].fill('')

    aliases = dict(_SHARED_ALIASES)
    aliases.update({key: (value,) for key, value in _DEEP_ALIASES[deep_catalog].items()})
    if deep_catalog == 'ukidss':
        aliases.update({'ukidss_' + key: (key,) for key in _UKIDSS_NATIVE})
    # First preference is an existing standardized column. The raw name is
    # used when building directly from the merger's output.
    for dest in STANDARD_DTYPE.names:
        if dest == 'hic_designation':
            continue
        candidates = (dest,) + aliases.get(dest, ())
        source = next((key for key in candidates if key in columns), None)
        if source is None:
            continue
        values = table[source]
        valid = ~np.ma.getmaskarray(values)
        if np.all(valid):
            out[dest] = np.asarray(values)
        elif np.any(valid):
            out[dest][valid] = np.asarray(values)[valid]

    native = np.char.lower(_as_text(out['native']))
    is_2mass = native == '2mass'
    expected = 'ukidss' if deep_catalog == 'ukidss' else 'vvv'
    is_deep = native == expected
    if np.any(~(is_2mass | is_deep)):
        i = np.flatnonzero(~(is_2mass | is_deep))[0]
        raise ValueError('Unexpected native={!r} at row {}'.format(native[i], i))

    mass_id = _as_text(out['designation'])
    deep_id = out['nir_sourceid']
    missing_id = (is_2mass & (mass_id == '')) | (is_deep & (deep_id == -1))
    if np.any(missing_id):
        raise ValueError('Missing source ID at row {}'.format(np.flatnonzero(missing_id)[0]))
    deep_text = deep_id.astype(str)
    names = np.where(is_2mass, np.char.add('2MASS:', mass_id),
                     np.char.add(deep_catalog.upper() + ':', deep_text))
    if np.any(np.char.str_len(names) > 33):
        raise ValueError('hic_designation exceeds U33; increase STANDARD_DTYPE')
    out['hic_designation'] = names

    # the jmag/hmag/kmag columns are the native values for both 2mass/deepnir
    # the original "_native" columns just have the deepnir native values
    out['jmag_native'] = table['jmag']
    out['hmag_native'] = table['hmag']
    out['kmag_native'] = table['kmag']
    # use the 2mass transformed mags as the default ones
    out['jmag'] = table['jmag_2mass']
    out['hmag'] = table['hmag_2mass']
    out['kmag'] = table['kmag_2mass']

    return out

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
    
    tablist = []
    numlist = []
    for i in range(len(querypix)):
        filename = basedir+'/hic_{:d}.fits'.format(querypix[i])
        if os.path.exists(filename):
            tab = Table.read(filename)

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
                
            # Standardize the columns and add HIC designation
            tab = build_hic_table(tab, deepnircatalog)
            tab['hpix'] = querypix[i]
        else:
            print(filename,'not found')
            continue
        tablist.append(tab)
        numlist.append(len(tab))
    hic = np.concatenate(tablist)
    
    # Only take the sources within the desired radius
    idx = cone_search(hic["ra"], hic["dec"], ra, dec, radius)
    hic = hic[idx]
    
    return hic
