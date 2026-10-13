import os
import numpy as np
#import numpy.lib.recfunctions as rfn
from astropy.table import Table,vstack
from astropy.coordinates import SkyCoord
from dlnpyutils import utils as dln,coords
import healpy as hp
from apogee_drp.database import apogeedb
from dl import queryClient as qc
import merge_2mass_vvv
import merge_2mass_ukidss
import traceback
from astropy.io import fits
import queryhic
from glob import glob

# Make the HGE Input Catalog (HIC)


def getdata(hpix,catalog,nside=256,radius=None,count=False,inpix=False):

    # Use sharded hpix file
    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/'
    if os.path.exists(basedir+catalog.lower()):
        if radius is not None:
            vec = hp.pix2vec(nside,hpix)
            querypix = hp.query_disc(256,vec,np.deg2rad(radius),inclusive=True)
        else:
            querypix = [hpix]
        tab = []
        for i in range(len(querypix)):
            pixfile = basedir+catalog.lower()+'/ring256/'+catalog.lower()+'_{:d}.fits'.format(int(querypix[i]))
            if os.path.exists(pixfile):
                tab1 = Table.read(pixfile)
                # might need to fix vvvx column names
                if catalog.lower()=='vvvx' and 'jmag' not in tab1.colnames:
                    tab1['jpsfmag'].name = 'jmag'
                    tab1['jpsfmagerr'].name = 'jerr'
                    tab1['hpsfmag'].name = 'hmag'
                    tab1['hpsfmagerr'].name = 'herr'
                    tab1['kspsfmag'].name = 'kmag'
                    tab1['kspsfmagerr'].name = 'kerr'
                elif catalog.lower()=='virac2' and 'ksmag' in tab1.colnames:
                    tab1['jmag'].name = 'jmag'
                    tab1['e_jmag'].name = 'jerr'
                    tab1['hmag'].name = 'hmag'
                    tab1['e_hmag'].name = 'herr'
                    tab1['ksmag'].name = 'kmag'
                    tab1['e_ksmag'].name = 'kerr'
                tab.append(tab1)
        if len(tab)>0:
            tab = vstack(tab)
        return tab
        
    ra,dec = hp.pix2ang(nside,hpix,lonlat=True)

    nestpix = hp.ring2nest(nside, hpix)
    whrh29 = 'healpix_29 >= {:d}::bigint * 4398046511104::bigint'.format(int(nestpix))
    whrh29 += ' AND healpix_29 <  {:d}::bigint * 4398046511104::bigint'.format(int(nestpix)+1)

    racol = 'ra'
    deccol = 'dec'
    if catalog.lower()=='2mass':
        deccol = 'decl'
        cols = 'ra,decl as dec,designation,j_m as jmag,j_cmsig as jerr,h_m as hmag,h_cmsig as herr,'
        cols += 'k_m as kmag,k_cmsig as kerr,ph_qual,rd_flg'
        table = 'catalogdb.twomass_psc'
    elif catalog.lower()=='virac2':
        deccol = 'de'
        cols = 'sourceid,ra,de as dec,phot_j_mean_mag as jmag,phot_j_std_mag as e_jmag,phot_h_mean_mag as hmag,'
        cols += 'phot_h_std_mag as e_hmag,phot_ks_mean_mag as ksmag,phot_ks_std_mag as e_ksmag,'
        cols += 'chisq,uwe'
        table = 'catalogdb.virac2'
    elif catalog.lower()=='ukidss':
        cols = 'sourceid,ra,dec,japermag3 as jmag,japermag3err as jerr,hapermag3 as hmag,hapermag3err as herr,'
        cols += 'kapermag3 as kmag,kapermag3err as kerr,'
        cols += 'jpperrbits, hpperrbits, kpperrbits,mergedclass, mergedclassstat, pstar,'
        cols += 'jclass, hclass, kclass, jell, hell, kell'
        #cols = '*'
        table = 'catalogdb.ukidss_dr11plus_gps'
    elif catalog.lower()=='glimpse':
        cols = 'ra,dec,designation,j_m,j_cmsig,h_m,h_cmsig,k_m,k_cmsig,ph_qual,rd_flg'
        cols = '*'
        table = 'catalogdb.glimpse'
    elif catalog.lower()=='apoglimpse':
        cols = 'ra,dec,designation,j_m,j_cmsig,h_m,h_cmsig,k_m,k_cmsig,ph_qual,rd_flg'
        cols = '*'
        table = 'catalogdb.apoglimpse'
    elif catalog.lower()=='deepglimpse':
        cols = 'ra,dec,designation,j_m,j_cmsig,h_m,h_cmsig,k_m,k_cmsig,ph_qual,rd_flg'
        cols = '*'
        table = 'catalogdb.deep_glimpse'
    elif catalog.lower()=='glimpseproper':
        cols = 'ra,dec,designation,j_m,j_cmsig,h_m,h_cmsig,k_m,k_cmsig,ph_qual,rd_flg'
        cols = '*'
        table = 'catalogdb.glimpse_proper'
    elif catalog.lower()=='allwise':
        cols = 'ra,dec,designation,j_m,j_cmsig,h_m,h_cmsig,k_m,k_cmsig,ph_qual,rd_flg'
        cols = '*'
        table = 'catalogdb.allwise'
    elif catalog.lower()=='catwise':
        cols = 'ra,dec,designation,j_m,j_cmsig,h_m,h_cmsig,k_m,k_cmsig,ph_qual,rd_flg'
        cols = '*'
        table = 'catalogdb.catwise'
    elif catalog.lower()=='unwise':
        cols = 'unwise_objid,ra,dec,primary_status,flux_w1,dflux_w1,flux_w2,dflux_w2,'
        cols += 'fracflux_w1, fracflux_w2, qf_w1, qf_w2, rchi2_w1, rchi2_w2,'
        cols += 'flags_unwise_w2, flags_info_w2'
        table = 'catalogdb.unwise'
    elif catalog.lower()=='gaiadr3':
        cols = 'source_id,ra,dec,pmra,pmdec,parallax,parallax_error,phot_g_mean_mag as gmag,'
        cols += 'phot_bp_mean_mag as bpmag,phot_rp_mean_mag as rpmag,ruwe,astrometr_excess_noise'
        table = 'catalogdb.gaia_dr3_source'
    elif catalog.lower()=='vhs':
        racol = 'ra2000'
        deccol = 'dec2000'
        cols = 'sourceid,ra2000 as ra,dec2000 as dec,japermag3 as jmag,japermag3err as jerr,'
        cols += 'hapermag3 as hmag,hapermag3err as herr,ksapermag3 as kmag,ksapermag3err as kerr,'
        cols += 'mergedclass'
        table = 'vhs_dr5.vhs_cat_v3'
    else:
        print(catalog,'not supported')
        return []
        
    if count:
        cols = 'count(*)'

    # Spatial query
    whr = None
    # use healpix index when possible
    catalogs_have_healpix29 = ['virac2','ukidss','apoglimpse','deepglimpse','glimpseproper']
    if radius is None and catalog.lower() in catalogs_have_healpix29:
        whr = whrh29
    if radius is None and catalog.lower()=='vhs':
        whr = 'ring256={:d}'.format(hpix)
    if whr is None and radius is None:
        resol = hp.nside2resol(nside,arcmin=True)/60
        radius = resol/1.4
        whr = 'q3c_radial_query({:},{:},{:},{:},{:})'.format(racol,deccol,ra,dec,radius)
        
    sql = 'select '+cols+' from '+table+' where '+whr
    
    if catalog.lower()!='vhs':
        db = apogeedb.DBSession('operations')
        tab = db.query(sql=sql)
        db.close()
    else:
        tab = qc.query(sql=sql,fmt='table')
        tab = np.array(tab)

    # add magnitudes for unWISE
    if catalog.lower()=='unwise':
        tab = Table(tab)
        tab['w1mag'] = np.nan
        tab['w1err'] = np.nan
        gd1, = np.where(tab['flux_w1']>0)
        tab['w1mag'][gd1] = 22.5 - 2.5*np.log10(tab['flux_w1'][gd1])
        tab['w1err'][gd1] = 1.085736 * tab['dflux_w1'][gd1] / tab['flux_w1'][gd1]
        tab['w2mag'] = np.nan
        tab['w2err'] = np.nan
        gd2, = np.where(tab['flux_w2']>0)
        tab['w2mag'][gd2] = 22.5 - 2.5*np.log10(tab['flux_w2'][gd2])
        tab['w2err'][gd2] = 1.085736 * tab['dflux_w2'][gd2] / tab['flux_w2'][gd2]
        tab = np.array(tab)

    if inpix and catalog.lower() not in catalogs_have_healpix29+['vhs']:
        pp = hp.ang2pix(nside,tab['ra'],tab['dec'],lonlat=True)
        gd, = np.where(pp == hpix)
        tab = tab[gd]
        
    return tab

def pixcoverage(pix,tab):
    """ healpix coverage fraction """
    if len(tab)==0:
        return 0.0
    vec = hp.boundaries(256, pix, step=1, nest=False)
    lon, lat = hp.vec2ang(vec.T, lonlat=True)
    # get 2D histogram of data
    out = np.histogram2d(tab['ra'],tab['dec'],range=[[np.min(lon),np.max(lon)],[np.min(lat),np.max(lat)]],bins=20)
    im,xedge,yedge = out
    x = xedge[:-1] + (xedge[1]-xedge[0])*0.5
    y = yedge[:-1] + (yedge[1]-yedge[0])*0.5
    xx,yy = np.meshgrid(x,y)
    # get pixels inside the healpix region
    ind,cutind = dln.roi_cut(lon,lat,xx.ravel(),yy.ravel())
    nall = len(cutind)
    ncov = np.sum(im > 0)
    frac = np.minimum(ncov/nall,1.0)
    return frac

def choose_nir_catalog(coverage, tolerance=0.10):
    """coverage: dict mapping catalog names to fractions in [0, 1]."""
    priority = ['virac2', 'vvvx', 'ukidss', 'vhs']

    fractions = np.array(
        [coverage.get(cat, np.nan) for cat in priority],
        dtype=float
    )
    valid = np.isfinite(fractions) & (fractions > 0)

    if not valid.any():
        return None

    best_coverage = fractions[valid].max()
    eligible = valid & (fractions >= best_coverage - tolerance)

    return priority[np.flatnonzero(eligible)[0]]

def getdeepnircatalog(hpix):
    """ Get single deep NIR catalog """

    virac2 = getdata(hpix,'virac2')
    nvirac2 = len(virac2)
    virac2frac = pixcoverage(hpix,virac2)
    vvvx = getdata(hpix,'vvvx')
    nvvvx = len(vvvx)
    vvvxfrac = pixcoverage(hpix,vvvx)
    ukidss = getdata(hpix,'ukidss')
    nukidss = len(ukidss)
    ukidssfrac = pixcoverage(hpix,ukidss)
    #vhs = getdata(hpix,'vhs')
    #nvhs = len(vhs)
    #vhsfrac = pixcoverage(hpix,vhs)
    vhs = []
    nvhs = 0
    vhsfrac = 0.0
    
    # Priority:
    # 1) virac2
    # 2) vvvx
    # 3) ukidss
    # 4) vhs

    coverage = dict(virac2=virac2frac, vvvx=vvvxfrac,
                    ukidss=ukidssfrac, vhs=vhsfrac)
    usecatalog = choose_nir_catalog(coverage)   # 'vvvx'
    if usecatalog is None:
        return []
    
    #number = np.array([nvirac2,nvvvx,nukidss,nvhs])
    #bestind = np.where(number > 0)[0]
    #if len(bestind)==0:
    #    print('no deep nir catalog for',hpix)
    #    return []
    #else:
    #    bestind = bestind[0]
    #usecatalog = ['virac2','vvvx','ukidss','vhs'][bestind]
    #tab = [virac2,vvvx,ukidss,vhs][bestind]

    tab = {'virac2':virac2,'vvvx':vvvx,'ukidss':ukidss,'vhs':vhs}[usecatalog]
    
    tab = Table(tab)
    tab['nircatalog'] = usecatalog
    print('using',usecatalog,len(tab))
    
    return tab

def combinenir(tmass,deepnir):
    """ Combine 2MASS and deep NIR catalog """

    # crossmatch and deep with photometric offsets
    catalog = deepnir['nircatalog'][0]
    
    # virac2
    if catalog=='virac2':
        # bright limit is H=12
        nir = merge_2mass_vvv.merge(tmass,deepnir)
    elif catalog=='vvvx':
        # bright limit is H=12
        nir = merge_2mass_vvv.merge(tmass,deepnir)
    elif catalog=='ukidss':
        # bright limit is H~11
        nir = merge_2mass_ukidss.merge(tmass,deepnir)
    elif catalog=='vhs':
        # bright limit is H~
        #import pdb; pdb.set_trace()
        raise Exception('merging vhs not supported')
    else:
        print('not supported',catalog)
        nir = []
        
    return nir

def add_mir_columns(tab):
    mirdtype = [('glimpse_designation','<U43'),('glimpse_mag36',np.float32),('glimpse_mag36err',np.float32),
                ('glimpse_mag45',np.float32),('glimpse_mag45err',np.float32),
                #('glimpse_rms36',np.float32),('glimpse_rms45',np.float32),('glimpse_dens36',np.float32),
                #('glimpse_dens45',np.float32),('glimpse_sqf36',int),('glimpse_sqf45',int),
                ('unwise_objid','S26'),('unwise_primary_status',int),('unwise_flux_w1',np.float32),('unwise_dflux_w1',np.float32),
                ('unwise_flux_w2',np.float32),('unwise_dflux_w2',np.float32),('unwise_fracflux_w1',np.float32),
                ('unwise_fracflux_w2',np.float32),('unwise_qf_w1',np.float32),('unwise_qf_w2',np.float32),
                ('unwise_rchi2_w1',np.float32),('unwise_rchi2_w2',np.float32),
                ('unwise_flags_unwise_w2',np.float32),('unwise_flags_info_w2',np.float32),('unwise_w1mag',np.float32),
                ('unwise_w1err',np.float32),('unwise_w2mag',np.float32),('unwise_w2err',np.float32),
                ('mircatalog','S10')]
    newdtype = np.dtype(tab.dtype.descr + mirdtype)
    newtab = np.zeros(len(tab),dtype=newdtype)
    for c in tab.dtype.names:
        newtab[c] = tab[c]
    for i in range(len(mirdtype)):
        c = mirdtype[i][0]
        typ = mirdtype[i][1]
        if typ == np.float32:
            newtab[c] = np.nan
        elif typ == int:
            newtab[c] = -1

    return newtab

def rjcederedden(tab):
    """ perform standard RJCE dereddening. """

    tab['ak'] = np.nan
    tab['ah'] = np.nan
    tab['ejk']= np.nan
    tab['jk0'] = np.nan
    tab['h0']= np.nan

    jcol = 'jmag'
    hcol = 'hmag'
    kcol = 'kmag'
    if 'jmag_2mass' in tab.colnames:
        jcol = 'jmag_2mass'
        hcol = 'hmag_2mass'
        kcol = 'kmag_2mass'
    
    # GLIMPSE
    if tab['mircatalog'][0]=='glimpse':
        gd, = np.where(np.isfinite(tab[hcol]) & (tab[hcol] < 50) & (tab[hcol]>0) &
                       np.isfinite(tab['glimpse_mag45']) & (tab['glimpse_mag45'] < 50) & (tab['glimpse_mag45']>0))
        tab['ak'][gd] = 0.918*(tab[hcol][gd] - tab['glimpse_mag45'][gd] - 0.08)
    # WISE
    else:
        gd, = np.where(np.isfinite(tab[hcol]) & (tab[hcol] < 50) & (tab[hcol]>0) &
                       np.isfinite(tab['unwise_w2mag']) & (tab['unwise_w2mag'] < 50) & (tab['unwise_w2mag']>0))
        tab['ak'][gd] = 0.918*(tab[hcol][gd] - tab['unwise_w2mag'][gd] - 0.05)
        
    tab['ah'][gd] = 1.55*tab['ak'][gd]
    tab['ejk'][gd] = 1.5*tab['ak'][gd]
    gdjhk, = np.where(np.isfinite(tab[jcol]) & (tab[jcol] < 50) & (tab[jcol]>0) &
                      np.isfinite(tab[hcol]) & (tab[hcol] < 50) & (tab[hcol]>0) &
                      np.isfinite(tab[kcol]) & (tab[kcol] < 50) & (tab[kcol]>0))
    tab['jk0'][gdjhk] = tab[jcol][gdjhk]-tab[kcol][gdjhk]-tab['ejk'][gdjhk]
    tab['h0'][gd] = tab[hcol][gd]-tab['ah'][gd]
    
    return tab

    
def makehicsingle(hpix,nside=256):
    """ Make HIC for single healpix """

    print('Making HIC for',hpix)
    
    # ---- NIR catalog ----
    tmass = getdata(hpix,'2mass')
    print('2MASS',len(tmass))
    
    # pick single deep NIR catalog to use
    deepnir = getdeepnircatalog(hpix)
    
    # Combine 2MASS and deep NIR catalog
    # deal with cross-matching and photometric offsets
    nir = combinenir(Table(tmass),Table(deepnir))

    # Make sure to save the name of the deep NIR catalog
    nir['deepnircatalog'] = deepnir['nircatalog'][0]

    
    # ---- MIR catalog ----

    # allglimpse
    glimpse = getdata(hpix,'allglimpse')

    # unwise
    unwise = getdata(hpix,'unwise')

    if len(glimpse)>0:
        mir = glimpse
        mircatalog = 'glimpse'
    else:
        mir = unwise
        mircatalog = 'unwise'

    # add MIR columns
    final = nir.copy()
    final = add_mir_columns(final)
        
    # Xcmatch
    mir_ind,nir_ind,dist = coords.xmatch(mir['ra'],mir['dec'],nir['ra'],nir['dec'],1.5,unique=True)

    if mircatalog=='glimpse':
        final['mircatalog'] = 'glimpse'
        outcols = ['glimpse_designation','glimpse_mag36','glimpse_mag36err','glimpse_mag45',
                   'glimpse_mag45err']
        #,'glimpse_rms36','glimpse_rms45','glimpse_dens36',
        #           'glimpse_dens45','glimpse_sqf36','glimpse_sqf45']
        incols = ['designation','mag3_6','d3_6m','mag4_5','d4_5m']
        #           'rms_f3_6','rms_f4_5','dens_3_6','dens_4_5','sqf_3_6','sqf_4_5']
        for i in range(len(outcols)):
            final[outcols[i]][nir_ind] = mir[incols[i]][mir_ind]
    else:
        final['mircatalog'] = 'unwise'
        outcols = ['unwise_objid','unwise_primary_status','unwise_flux_w1','unwise_dflux_w1',
		   'unwise_flux_w2','unwise_dflux_w2','unwise_fracflux_w1',
                   'unwise_fracflux_w2','unwise_qf_w1','unwise_qf_w2',
                   'unwise_rchi2_w1','unwise_rchi2_w2',
                   'unwise_flags_unwise_w2','unwise_flags_info_w2','unwise_w1mag',
		   'unwise_w1err','unwise_w2mag','unwise_w2err']
        incols = ['unwise_objid','primary_status','flux_w1','dflux_w1','flux_w2','dflux_w2',
                  'fracflux_w1','fracflux_w2','qf_w1','qf_w2','rchi2_w1','rchi2_w2',
                  'flags_unwise_w2','flags_info_w2','w1mag','w1err','w2mag','w2err']
        for i in range(len(outcols)):
            final[outcols[i]][nir_ind] = mir[incols[i]][mir_ind]

    # standardize and add hic_designation
    final = queryhic.build_hic_table(final, deepnir['nircatalog'][0])
    final = Table(final)
    final['deepnircatalog'] = deepnir['nircatalog'][0]
    final['hpix'] = hpix

    # add standard RJCE dereddening
    final = rjcederedden(Table(final))
        
    print(len(final),'final sources')
    
    return final

    
def makehic(clobber=False):
    """ Make the HGE Input Catalog (HIC) """

    # (2MASS + VIRAC2 + VVVX + UKIDSS) + allglimpse + unWISE

    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/hic/ring256/'
    allpix = dln.readlines('allpix.txt')
    allpix = np.array(allpix).astype(int)
      
    for i in range(len(allpix)):
        print(i+1,allpix[i])
        outfile = basedir+'/hic_{:d}.fits'.format(allpix[i])
        if os.path.exists(outfile) and clobber==False:
            print(outfile,'exists')
            continue
        ra,dec = hp.pix2ang(256,allpix[i],lonlat=True)
        coo = SkyCoord(ra,dec,unit='deg',frame='icrs')
        glon = coo.galactic.l.degree
        #if glon > 20:
        #    print('skipping ukidss for now')
        #    continue
        try:
            tab = makehicsingle(allpix[i])
            #tab = rjcederedden(Table(tab))
            print('Writing to',outfile)
            hdulist = fits.HDUList()
            hdulist.append(fits.table_to_hdu(tab))
            hdulist[0].header['pix'] = allpix[i]
            hdulist[0].header['dpnircat'] = tab['deepnircatalog'][0]
            hdulist[0].header['mircat'] = tab['mircatalog'][0]
            gd, = np.where(np.isfinite(tab['ak']))
            if len(gd)>0:
                medak = np.nanmedian(tab['ak'][gd])
                ak95 = np.percentile(tab['ak'][gd],95)
            else:
                medak = 'NaN'
                ak95 = 'NaN'
            hdulist[0].header['medak'] = medak
            hdulist[0].header['ak95'] = ak95
            hdulist.writeto(outfile,overwrite=True)
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()

        #import pdb; pdb.set_trace()
            
        print('')

def addext():
    """ go through all of the hic files and add extinctions """
    # and metadata in header

    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/hic/ring256/'
    allpix = dln.readlines('allpix.txt')
    allpix = np.array(allpix).astype(int)
      
    for i in range(len(allpix)):
        print(i+1,allpix[i])
        outfile = basedir+'/hic_{:d}.fits'.format(allpix[i])
        ra,dec = hp.pix2ang(256,allpix[i],lonlat=True)
        coo = SkyCoord(ra,dec,unit='deg',frame='icrs')
        glon = coo.galactic.l.degree
        #if glon > 20:
        #    print('skipping ukidss for now')
        #    continue
        try:
            tab = Table.read(outfile)
            deepnir = getdeepnircatalog(allpix[i])
            tab['deepnircatalog'] = deepnir['nircatalog'][0]
            tab = rjcederedden(tab)
            print('Writing to',outfile)
            hdulist = fits.HDUList()
            hdulist.append(fits.table_to_hdu(tab))
            hdulist[0].header['pix'] = allpix[i]
            hdulist[0].header['dpnircat'] = tab['deepnircatalog'][0]
            hdulist[0].header['mircat'] = tab['mircatalog'][0]
            gd, = np.where(np.isfinite(tab['ak']))
            if len(gd)>0:
                medak = np.nanmedian(tab['ak'][gd])
                ak95 = np.percentile(tab['ak'][gd],95)
            else:
                medak = 'NaN'
                ak95 = 'NaN'
            hdulist[0].header['medak'] = medak
            hdulist[0].header['ak95'] = ak95
            hdulist.writeto(outfile,overwrite=True)
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            
        print('')


def standardizeall():
    """ go through all of the hic files and add extinctions """
    # and metadata in header

    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/hic/ring256/'
    allpix = dln.readlines('allpix.txt')
    allpix = np.array(allpix).astype(int)
      
    for i in range(len(allpix)):
        print(i+1,allpix[i])
        outfile = basedir+'/hic_{:d}.fits'.format(allpix[i])
        ra,dec = hp.pix2ang(256,allpix[i],lonlat=True)
        coo = SkyCoord(ra,dec,unit='deg',frame='icrs')
        glon = coo.galactic.l.degree
        #if glon > 20:
        #    print('skipping ukidss for now')
        #    continue
        try:
            tab = Table.read(outfile)
            if 'hic_designation' in tab.colnames:
                print('already standardized')
                continue
            
            # standardize and add hic_designation
            if 'hell' in tab.colnames:
                deepnircatalog = 'ukidss'
            elif 'uwe' in tab.colnames:
                deepnircatalog = 'virac2'
            elif 'jmhpnt' in tab.colnames:
                deepnircatalog = 'vvvx'
            else:
                raise Exception('do not know this type')
            tab = queryhic.build_hic_table(tab,deepnircatalog)
            tab['hpix'] = allpix[i]
            tab = rjcederedden(Table(tab))
            print('Writing to',outfile)
            hdulist = fits.HDUList()
            hdulist.append(fits.table_to_hdu(tab))
            hdulist[0].header['pix'] = allpix[i]
            hdulist[0].header['dpnircat'] = deepnircatalog
            hdulist[0].header['mircat'] = tab['mircatalog'][0]
            gd, = np.where(np.isfinite(tab['ak']))
            if len(gd)>0:
                medak = np.nanmedian(tab['ak'][gd])
                ak95 = np.percentile(tab['ak'][gd],95)
            else:
                medak = 'NaN'
                ak95 = 'NaN'
            hdulist[0].header['medak'] = medak
            hdulist[0].header['ak95'] = ak95
            hdulist.writeto(outfile,overwrite=True)
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            
        print('')

def summary():

    files = glob('ring256/hic_*.fits')
    files.sort()
    dt = [('filename',str,60),('pix',int),('ra',float),('dec',float),
          ('l',float),('b',float),('nsources',int),
          ('deepnircatalog',str,30),('mircatalog',str,20),
          ('medak',float),('ak95',float)]
    sumtab = np.zeros(len(files),dtype=np.dtype(dt))
    for i in range(len(files)):
        print(i+1,files[i])
        hdu = fits.open(files[i])
        sumtab['filename'][i] = files[i]
        sumtab['pix'][i] = hdu[0].header['pix']
        sumtab['nsources'][i] = hdu[1].header['naxis2']
        sumtab['deepnircatalog'][i] = hdu[0].header['dpnircat']
        sumtab['mircatalog'][i] = hdu[0].header['mircat']
        sumtab['medak'][i] = hdu[0].header['medak']
        sumtab['ak95'][i] = hdu[0].header['ak95']
        hdu.close()

    ra,dec = hp.pix2ang(256,sumtab['pix'],lonlat=True)
    sumtab['ra'] = ra
    sumtab['dec'] = dec
    coo = SkyCoord(ra,dec,unit='deg',frame='icrs')
    sumtab['l'] = coo.galactic.l.degree
    sumtab['b'] = coo.galactic.b.degree
        
    Table(sumtab).write('hic_summary.fits',overwrite=True)

    import pdb; pdb.set_trace()
