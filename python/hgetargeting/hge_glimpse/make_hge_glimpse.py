import os
import numpy as np
from astropy.table import Table,vstack
from astropy.coordinates import SkyCoord
from dlnpyutils import utils as dln,coords
import healpy as hp
from apogee_drp.database import apogeedb
from dl import queryClient as qc
import traceback
from astropy.io import fits
from glob import glob

# Make the HGE GLIMPSE catalog



def getdata(hpix,catalog,nside=256,count=False,boundary=False,verbose=True):

    if boundary != False and boundary != None:
        if verbose:
            print(hpix,catalog,'with boundary')
        # query neighbors too
        neipix = hp.get_all_neighbours(nside,hpix)
        pix = [hpix]+list(neipix)
        tab = []
        for i in range(len(pix)):
            tab1 = getdata(pix[i],catalog,nside=nside,boundary=False)
            if len(tab1)>0:
                tab.append(Table(tab1))
        tab = vstack(tab)
        # impose boundary
        vecbound = hp.boundaries(nside,hpix,step=100)
        rabound, decbound = hp.vec2ang(np.transpose(vecbound),lonlat=True)
        # Expand the boundary by the buffer size
        cenra, cendec = hp.pix2ang(nside,hpix,lonlat=True)
        # reproject onto tangent plane
        lonbound, latbound = coords.rotsphcen(rabound,decbound,cenra,cendec,gnomic=True)
        # expand by a fraction, it's not an extact boundary but good enough
        if boundary != True:
            buffsize = float(boundary)/3600. # in deg
        else:
            buffsize = 20.0/3600. # in deg
        radbound = np.sqrt(lonbound**2+latbound**2)
        frac = 1.0 + 1.5*np.max(buffsize/radbound)
        lonbuff = lonbound*frac
        latbuff = latbound*frac
        rabuff, decbuff = coords.rotsphcen(lonbuff,latbuff,cenra,cendec,gnomic=True,reverse=True)
        if (np.max(rabuff)-np.min(rabuff))>100:  # deal with RA=0 wraparound
            bd, = np.where(rabuff>180)
            if len(bd)>0:
                rabuff[bd] -=360.0
        # Get all sources inside the boundary
        ll,bb = coords.rotsphcen(tab['ra'],tab['dec'],cenra,cendec,gnomic=True)
        ind, cutind = dln.roi_cut(lonbuff,latbuff,ll,bb)
        tab = tab[cutind]
        if verbose:
            print(len(tab),'inside boundary')
        return tab
        
    ra,dec = hp.pix2ang(nside,hpix,lonlat=True)
    nestpix = hp.ring2nest(nside, hpix)
    whrh29 = 'healpix_29 >= {:d}::bigint * 4398046511104::bigint'.format(int(nestpix))
    whrh29 += ' AND healpix_29 <  {:d}::bigint * 4398046511104::bigint'.format(int(nestpix)+1)
    cols = '*'
    if count:
        cols = 'count(*)'
    sql = 'select '+cols+' from '+catalog+' where '+whrh29
    
    db = apogeedb.DBSession('as5')
    tab = db.query(sql=sql)
    db.close()
    if verbose:
        print(hpix,catalog,len(tab))
    
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

def standardize(data,catalog):
    """ Standardize columns """
    data = Table(data)
    if catalog=='glimpse':
        pass
    elif catalog=='apoglimpse':
        data['survey'] = 'APOGLIMPSE'
    elif catalog=='deep_glimpse':
        data['survey'] = 'DEEP GLIMPSE'
    elif catalog=='glimpse_proper':
        data['survey'] = 'GLIMPSE PROPER'
    else:
        raise Exception(catalog+' not supported')
    #data = np.array(data)
    cols = ['designation', 'tmass_designation', 'tmass_cntr', 'l',
            'b', 'dl', 'db', 'ra', 'dec', 'dra', 'ddec', 'csf', 'mag_j',
            'dj_m', 'mag_h', 'dh_m', 'mag_ks', 'dks_m', 'mag3_6', 'd3_6m',
            'mag4_5', 'd4_5m', 'mag5_8', 'd5_8m', 'mag8_0', 'd8_0m', 'f_j',
            'df_j', 'f_h', 'df_h', 'f_ks', 'df_ks', 'f3_6', 'df3_6', 'f4_5',
            'df4_5', 'f5_8', 'df5_8', 'f8_0', 'df8_0', 'rms_f3_6', 'rms_f4_5',
            'rms_f5_8', 'rms_f8_0', 'sky_3_6', 'sky_4_5', 'sky_5_8',
            'sky_8_0', 'sn_j', 'sn_h', 'sn_ks', 'sn_3_6', 'sn_4_5', 'sn_5_8',
            'sn_8_0', 'dens_3_6', 'dens_4_5', 'dens_5_8', 'dens_8_0', 'm3_6',
            'm4_5', 'm5_8', 'm8_0', 'n3_6', 'n4_5', 'n5_8', 'n8_0', 'sqf_j',
            'sqf_h', 'sqf_ks', 'sqf_3_6', 'sqf_4_5', 'sqf_5_8', 'sqf_8_0',
            'mf3_6', 'mf4_5', 'mf5_8', 'mf8_0', 'survey', 'healpix_29']
    data = data[cols]
    # should check types
    return data
    
def mergesingle(hpix,nside=256):
    """ Make HGE-GLIMPSE for single healpix """

    print('Making HGE-GLIMPSE for',hpix)

    #catnames = ['glimpse','apoglimpse','deep_glimpse','glimpse_proper']
    catnames = ['deep_glimpse','apoglimpse','glimpse']
    dt = [('catalog',str,30),('num',int),('numbuff',int)]
    catalogs = Table(np.zeros(len(catnames),dtype=np.dtype(dt)))
    catalogs['catalog'] = catnames
    for i in range(len(catalogs)):
        res = getdata(hpix,catalogs['catalog'][i],count=True,verbose=False)[0][0]
        catalogs['num'][i] = res
        print(catalogs['catalog'][i],res)
    gd, = np.where(catalogs['num']>0)
    if len(gd)==0:
        print('No glimpse catalogs for this healpix')
        return []
    catalogs = catalogs[gd]
    
    print('')
    print(len(gd),'catalogs have data in this healpix')
    print(catalogs['catalog'].data)
    
    # Get the catalogs
    print('')
    print('Getting data')
    catalogdata = {}
    for i in range(len(catalogs)):
        if len(catalogs)==1:
            boundary = False
        else:
            boundary = True
        data = getdata(hpix,catalogs['catalog'][i],boundary=boundary)
        # standardize columns
        data = standardize(data,catalogs['catalog'][i])
        catalogdata[catalogs['catalog'][i]] = data
        catalogs['numbuff'][i] = len(data)
        
    # Xmatch
    if len(catalogs)>1:
        print('Cross-matching')
        dcr = 1.0
        allcatalog = catalogdata[catalogs['catalog'][0]]
        print('Starting with',catalogs['catalog'][0],len(allcatalog))
        for i in range(1,len(catalogs)):
            newcatalog = catalogdata[catalogs['catalog'][i]]
            ind1,ind2,dist = coords.xmatch(allcatalog['ra'],allcatalog['dec'],
                                           newcatalog['ra'],newcatalog['dec'],dcr,unique=True)
            print(catalogs['catalog'][i],len(ind1),'matches')
            if len(ind2) < len(newcatalog):
                left = np.delete(np.arange(len(newcatalog)),ind2)
                print(len(left),'left')
                allcatalog = vstack((allcatalog,newcatalog[left]))
        print(len(allcatalog),'sources')
    else:
        print('Notching to crossmatch')
        allcatalog = catalogdata[catalogs['catalog'][0]]
        
    # Now trim to the healpix
    allcatpix = hp.ang2pix(nside,allcatalog['ra'],allcatalog['dec'],lonlat=True)
    keep, = np.where(allcatpix==hpix)
    print(len(keep),'inside the healpix')
    allcatalog = allcatalog[keep]
        
    return allcatalog

    
def makehgeglimpse(clobber=False):
    """ Make the HGE GLIMPSE catalog """

    # Merge GLIMPSE, Deep GLIMPSE and APOGLIMPSE
    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/hge_glimpse/ring256/'
    allpix = dln.readlines('allpix.txt')
    allpix = np.array(allpix).astype(int)
      
    for i in range(len(allpix)):
        print('----',i+1,allpix[i],'----')
        outfile = basedir+'/hge_glimpse_{:d}.fits'.format(allpix[i])
        if os.path.exists(outfile) and clobber==False:
            print(outfile,'exists')
            continue
        ra,dec = hp.pix2ang(256,allpix[i],lonlat=True)
        coo = SkyCoord(ra,dec,unit='deg',frame='icrs')
        glon = coo.galactic.l.degree
        try:
            tab = mergesingle(allpix[i])
            if len(tab)==0:
                continue
            print('Writing to',outfile)
            hdulist = fits.HDUList()
            hdulist.append(fits.table_to_hdu(tab))
            hdulist[0].header['pix'] = allpix[i]
            hdulist[0].header['nsources'] = len(tab)
            usurvey = np.unique(tab['survey'].data)
            index = dln.create_index(tab['survey'].data)
            for j in range(len(index['value'])):
                hdulist[0].header.add_history('{:s}: {:d} sources'.format(index['value'][j],index['num'][j]))
            hdulist.writeto(outfile,overwrite=True)
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()

        #import pdb; pdb.set_trace()
            
        print('')

def summary():
    """ Make summary file for HGE GLIMPSE catalog """
    
    files = glob('ring256/hge_glimpse_*.fits')
    files.sort()
    dt = [('filename',str,60),('pix',int),('ra',float),('dec',float),
          ('l',float),('b',float),('nsources',int),
          ('ndeep',int),('napo',int),('nglimpse',int)]
    sumtab = np.zeros(len(files),dtype=np.dtype(dt))
    for i in range(len(files)):
        print(i+1,files[i])
        hdu = fits.open(files[i])
        sumtab['filename'][i] = files[i]
        sumtab['pix'][i] = hdu[0].header['pix']
        sumtab['nsources'][i] = hdu[1].header['naxis2']
        head = hdu[0].header.tostring(sep='\n').split('\n')
        srchd = [h for h in head if h.find('sources')>-1]
        for j in range(len(srchd)):
            srchd1 = srchd[j]
            catalog = srchd1[8:].split(':')[0]
            nsources = srchd1[8:].split(':')[1].strip().split()[0]
            if catalog=='DEEP GLIMPSE':
                sumtab['ndeep'][i] = int(nsources)
            elif catalog=='APOGLIMPSE':
                sumtab['napo'][i] = int(nsources)
            else:
                sumtab['nglimpse'][i] = int(nsources)
        #deephd = [h for h in head if h.find('DEEP')>-1]
        #apohd = [h for h in head if h.find('APO')>-1]
        #glmhd = [h for h in head if h.find('GLIMPSE I')>-1]
        #glmhd = [h for h in head if h.find('GLIMPSE I')>-1]
        hdu.close()

    ra,dec = hp.pix2ang(256,sumtab['pix'],lonlat=True)
    sumtab['ra'] = ra
    sumtab['dec'] = dec
    coo = SkyCoord(ra,dec,unit='deg',frame='icrs')
    sumtab['l'] = coo.galactic.l.degree
    sumtab['b'] = coo.galactic.b.degree
        
    Table(sumtab).write('hge_glimpse_summary.fits',overwrite=True)

    import pdb; pdb.set_trace()
