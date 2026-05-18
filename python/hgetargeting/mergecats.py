import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from dlnpyutils import utils as dln,coords,plotting as pl
from astropy.table import Table
from astropy.coordinates import SkyCoord

def derivephotcalib():
    """ Photometrically calibrate VVV photometry to 2MASS system """

    virac = Table.read('virac2_l340_b1.0_rad1.0.fits')
    virac = virac.filled()

    tmass = Table.read('tmass_l340_b1.0_rad1.0.fits')
    
    #o=pl.hist2d(virac['jmag']-virac['ksmag'],virac['hmag'],xr=[-1,5],yr=[20,5],log=True)
    #chopped off at H~12
    #is this in 2MASS bands or native VIRCAM bands

    #In [71]: len(vvv),len(virac),len(ind1)
    #Out[71]: (6208347, 3486176, 3162034)
    
    #In [72]: vvv2=vvv[ind1]
    #In [73]: virac2=virac[ind2]

    #o=pl.hist2d(vvv2['jmag'],virac2['jmag'],log=True)
    #plt.plot([10,20],[10,20],c='r')

    #o=pl.hist2d(vvv2['jmag'],vvv2['jmag']-virac2['jmag'],yr=[-2,2],log=True)
    #plt.axhline(0,c='r')

    #o=pl.hist2d(vvv2['hmag'],vvv2['hmag']-virac2['hmag'],yr=[-2,2],log=True)
    #plt.axhline(0,c='r')

    #o=pl.hist2d(vvv2['kmag'],vvv2['kmag']-virac2['ksmag'],yr=[-2,2],log=True)
    #plt.axhline(0,c='r')

    #np.nanmedian(vvv2['kmag']-virac2['ksmag'])
    #-0.022300

    #more like -0.05

    #o=pl.hist2d(virac['jmag']-virac['ksmag'],virac['hmag'],xr=[-1,8],yr=[19,10],log=True)

    #o=pl.hist2d(virac2['jmag']-virac2['ksmag'],virac2['hmag'],xr=[-1,8],yr=[19,10],log=True)

    #o=pl.hist2d(vvv['jmag']-vvv['kmag'],vvv['hmag'],xr=[-1,8],yr=[19,10],log=True)

    #o=pl.hist2d(vvv2['jmag']-vvv2['kmag'],vvv2['hmag'],xr=[-1,8],yr=[19,10],log=True)

    #I think virac2 has that missing area and is going brighter than the chang+2019 data (which I'm calling vvv)
    #but the change+2019 catalog goes deeper

    #combine virac2+2mass

    ind1,ind2,dist = coords.xmatch(tmass['ra'],tmass['dec'],virac['ra'],virac['dec'],1.5,unique=True)

    #In [101]: len(ind1)
    #Out[101]: 345683

    #In [102]: len(tmass)
    #Out[102]: 360633

    tmass3 = tmass[ind1]
    virac3 = virac[ind2]

    #o=pl.hist2d(tmass3['jmag']-tmass3['kmag'],tmass3['hmag'],xr=[-2,8],yr=[19,5],log=True)

    #o=pl.hist2d(virac3['jmag']-virac3['ksmag'],virac3['hmag'],xr=[-2,8],yr=[19,5],log=True)


    #ind3,ind4,dist = coords.xmatch(tmass['ra'],tmass['dec'],virac['ra'],virac['dec'],1.0,unique=True)
    #not much difference

    gd1, = np.where(tmass['hmag']<12.5)
    gd2, = np.where(virac['hmag']>=12.5)

    jmag = np.concatenate((tmass['jmag'][gd1],virac['jmag'][gd2]))
    hmag = np.concatenate((tmass['hmag'][gd1],virac['hmag'][gd2]))
    kmag = np.concatenate((tmass['kmag'][gd1],virac['ksmag'][gd2]))

    #o=pl.hist2d(jmag-kmag,hmag,xr=[-2,8],yr=[19,5],log=True)



    #o=pl.hist2d(tmass3['jmag'],tmass3['jmag']-virac3['jmag'],yr=[-2,2],log=True)
    #plt.axhline(0,c='r')

    #need to apply photometric transformations to get onto the 2MASS system

    ## Calibrate Jmag
    gdj, = np.where((np.abs(tmass3['jmag']-virac3['jmag']) < 0.3) & (virac3['jmag']>12) & (virac3['jmag']<15.5))
    #o=pl.hist2d(virac3['jmag'][gdj],tmass3['jmag'][gdj]-virac3['jmag'][gdj],yr=[-1,1],log=True)
    #plt.axhline(0,c='r')

    #o=pl.hist2d(virac3['jmag'][gdj]-virac3['ksmag'][gdj],tmass3['jmag'][gdj]-virac3['jmag'][gdj],yr=[-0.5,0.5],log=True)
    #plt.axhline(0,c='r')
    jcoef = np.polyfit(virac3['jmag'][gdj]-virac3['ksmag'][gdj],tmass3['jmag'][gdj]-virac3['jmag'][gdj],1)
    #x=np.linspace(0,4)
    #plt.plot(x,np.polyval(jcoef,x),c='blue')
    # array([ 0.02457057, -0.03392609])
    jmagcorr = np.polyval(jcoef,virac3['jmag']-virac3['ksmag'])+virac3['jmag']

    #o=pl.hist2d(virac3['jmag'][gdj]-virac3['ksmag'][gdj],tmass3['jmag'][gdj]-jmagcorr[gdj],yr=[-0.5,0.5],log=True)
    #plt.axhline(0,c='r')

    #In [164]: np.nanmedian(tmass3['jmag'][gdj]-jmagcorr[gdj])
    #Out[164]: 0.010979641435087117

    #o=pl.hist2d(virac3['jmag'][gdj]-virac3['ksmag'][gdj],tmass3['jmag'][gdj]-(jmagcorr[gdj]+0.011),yr=[-0.5,0.5],log=True)
    #plt.axhline(0,c='r')
    # that looks good

    #jcoef = np.array([ 0.02457057, -0.03392609])
    jcoef2 = np.array([ 0.02457057, -0.02294644])
    jmagcorr2 = np.polyval(jcoef2,virac3['jmag']-virac3['ksmag'])+virac3['jmag']

    #o=pl.hist2d(virac3['jmag'][gdj]-virac3['ksmag'][gdj],tmass3['jmag'][gdj]-jmagcorr2[gdj],yr=[-0.5,0.5],log=True)
    #plt.axhline(0,c='r')
    # that looks good

    virac3['jmagcorr'] = jmagcorr2

    ## Calibrate Hmag
    gdh, = np.where((np.abs(tmass3['hmag']-virac3['hmag']) < 0.3) & (virac3['hmag']>12) & (virac3['hmag']<15.5))
    #o=pl.hist2d(virac3['jmag'][gdh]-virac3['ksmag'][gdh],tmass3['hmag'][gdh]-virac3['hmag'][gdh],xr=[-1,5],yr=[-0.5,0.5],log=True)
    #plt.axhline(0,c='r')
    hcoef = np.polyfit(virac3['jmag'][gdh]-virac3['ksmag'][gdh],tmass3['hmag'][gdh]-virac3['hmag'][gdh],1)
    #x=np.linspace(0,4)
    #plt.plot(x,np.polyval(hcoef,x),c='blue')
    # array([-3.78361547e-22, -4.03011378e-02])

    from scipy.stats import binned_statistic
    out = binned_statistic(virac3['jmag'][gdh]-virac3['ksmag'][gdh],tmass3['hmag'][gdh]-virac3['hmag'][gdh],bins=np.linspace(-1,5,51),statistic='median')
    res,xedge,_ = out
    #plt.plot(xedge[:-1],res)
    xx = xedge[:-1]+(xedge[1]-xedge[0])*0.5
    gdx, = np.where(np.isfinite(res))
    hcoef2 = np.polyfit(xx[gdx],res[gdx],1)
    # array([-0.00813708, -0.0224336 ])
    #x=np.linspace(0,4)
    #plt.plot(x,np.polyval(hcoef2,x),c='blue')
    # looks good
    hmagcorr = np.polyval(hcoef2,virac3['jmag']-virac3['ksmag'])+virac3['hmag']

    #o=pl.hist2d(virac3['jmag'][gdh]-virac3['ksmag'][gdh],tmass3['hmag'][gdh]-hmagcorr[gdh],yr=[-0.5,0.5],log=True)
    #plt.axhline(0,c='r')
    # that looks good

    out2 = binned_statistic(virac3['jmag'][gdh]-virac3['ksmag'][gdh],tmass3['hmag'][gdh]-hmagcorr[gdh],bins=np.linspace(-1,5,51),statistic='median')
    res2,xedge2,_ = out2
    #plt.plot(xedge2[:-1],res2,c='blue')
    # looks good

    virac3['hmagcorr'] = hmagcorr


    ## Calibrate Kmag
    gdk, = np.where((np.abs(tmass3['kmag']-virac3['ksmag']) < 0.3) & (virac3['ksmag']>12) & (virac3['ksmag']<15.5))
    o=pl.hist2d(virac3['jmag'][gdk]-virac3['ksmag'][gdk],tmass3['kmag'][gdk]-virac3['ksmag'][gdk],xr=[-1,5],yr=[-0.5,0.5],log=True)
    #plt.axhline(0,c='r')
    kcoef = np.polyfit(virac3['jmag'][gdk]-virac3['ksmag'][gdk],tmass3['kmag'][gdk]-virac3['ksmag'][gdk],1)
    #x=np.linspace(0,4)
    #plt.plot(x,np.polyval(kcoef,x),c='blue')
    # array([-3.78361547e-22, -4.03011378e-02])

    from scipy.stats import binned_statistic
    out = binned_statistic(virac3['jmag'][gdk]-virac3['ksmag'][gdk],tmass3['kmag'][gdk]-virac3['ksmag'][gdk],bins=np.linspace(-1,5,51),statistic='median')
    res,xedge,_ = out
    #plt.plot(xedge[:-1],res)
    xx = xedge[:-1]+(xedge[1]-xedge[0])*0.5
    gdx, = np.where(np.isfinite(res))
    kcoef2 = np.polyfit(xx[gdx],res[gdx],1)
    # array([-0.00813708, -0.0224336 ])
    x=np.linspace(0,4)
    #plt.plot(x,np.polyval(kcoef2,x),c='blue')
    # looks good
    kmagcorr = np.polyval(kcoef2,virac3['jmag']-virac3['ksmag'])+virac3['ksmag']

    #o=pl.hist2d(virac3['jmag'][gdk]-virac3['ksmag'][gdk],tmass3['kmag'][gdk]-kmagcorr[gdk],yr=[-0.5,0.5],log=True)
    #plt.axhline(0,c='r')
    # that looks good

    out2 = binned_statistic(virac3['jmag'][gdk]-virac3['ksmag'][gdk],tmass3['kmag'][gdk]-kmagcorr[gdk],bins=np.linspace(-1,5,51),statistic='median')
    res2,xedge2,_ = out2
    #plt.plot(xedge2[:-1],res2,c='blue')
    # looks good

    virac3['kmagcorr'] = kmagcorr

def calibratevirac(virac):
    """ Photometrically calibrate VIRAC2 photometry to 2MASS system """

    jcoef = np.array([ 0.02457057, -0.02294644])
    hcoef = np.array([-0.00813708, -0.0224336 ])
    kcoef = np.array([-0.00813708, -0.0224336 ])

    jk = virac['jmag']-virac['ksmag']
    gd, = np.where(np.isfinite(virac['jmag']) &
                   np.isfinite(virac['ksmag']) &
                   (virac['jmag']<1e10) & (virac['ksmag']<1e10))
    if len(gd)<len(virac):
        bd = np.arange(len(virac))
        bd = np.delete(bd,gd)
        medjk = np.median(jk[gd])
        jk[bd] = medjk

    virac['jmagcorr'] = np.nan
    virac['hmagcorr'] = np.nan
    virac['kmagcorr'] = np.nan
        
    gdj, = np.where(np.isfinite(virac['jmag']) & (virac['jmag']<1e10))
    virac['jmagcorr'][gdj] = np.polyval(jcoef,jk[gdj])+virac['jmag'][gdj]
    gdh, = np.where(np.isfinite(virac['hmag']) & (virac['hmag']<1e10))
    virac['hmagcorr'][gdh] = np.polyval(hcoef,jk[gdh])+virac['hmag'][gdh]
    gdk, = np.where(np.isfinite(virac['ksmag']) & (virac['ksmag']<1e10))
    virac['kmagcorr'][gdk] = np.polyval(kcoef,jk[gdk])+virac['ksmag'][gdk]

    virac['jmag'].name = 'virac2_jmag'
    virac['hmag'].name = 'virac2_hmag'
    virac['ksmag'].name = 'virac2_ksmag'
    virac['jmagcorr'].name = 'jmag'
    virac['hmagcorr'].name = 'hmag'
    virac['kmagcorr'].name = 'kmag'
    virac['kserr'].name = 'kerr'
    
    return virac
    
def mergetmass(virac,tmass):
    """ Merge VIRAC2 and TMASS catalogs """

    # This assumes that the VIRAC photometry has already been calibrated to the 2MASS system
    
    gdt, = np.where(tmass['hmag']<12.5)
    gdv, = np.where(virac['hmag']>=12.5)

    ra = np.concatenate((tmass['ra'][gdt],virac['ra'][gdv]))
    #dec = np.concatenate((tmass['dec'][gdt],virac['dec'][gdv]))
    #jmag = np.concatenate((tmass['jmag'][gdt],virac['jmagcorr'][gdv]))
    #hmag = np.concatenate((tmass['hmag'][gdt],virac['hmagcorr'][gdv]))
    #kmag = np.concatenate((tmass['kmag'][gdt],virac['kmagcorr'][gdv]))
    
    #o=pl.hist2d(jmag-kmag,hmag,xr=[-2,8],yr=[19,5],log=True)

    #o=pl.hist2d(virac['jmagcorr']-virac['kmagcorr'],virac['hmagcorr'],xr=[-2,8],yr=[19,5],log=True)
    #plt.axhline(12.6,c='r')

    #o=pl.hist2d(tmass['jmag']-tmass['kmag'],tmass['hmag'],xr=[-2,8],yr=[19,5],log=True)
    #plt.axhline(12.6,c='r')

    print('Merging',len(gdt),'2MASS and',len(gdv),'VIRAC2 sources')
    
    dt = [('srcid',int),('ra',float),('dec',float),('plx',float),('e_plx',float),('pmra',float),
          ('e_pmra',float),('pmde',float),('e_pmde',float),('chi2',float),('uwe',float),
          ('zmag',float),('zerr',np.float32),('ymag',float),('yerr',np.float32),('jmag',float),
          ('jerr',np.float32),('hmag',float),('herr',np.float32),('kmag',float),('kerr',np.float32)]
    vtmass = np.zeros(len(ra),dtype=np.dtype(dt))
    for i in range(1,len(dt)):
        vtmass[vtmass.dtype.names[i]] = np.nan
    vtmass['srcid'][:len(gdt)] = np.arange(len(gdt))+1
    vtmass['ra'][:len(gdt)] = tmass['ra'][gdt]
    vtmass['dec'][:len(gdt)] = tmass['dec'][gdt]
    vtmass['jmag'][:len(gdt)] = tmass['jmag'][gdt]
    vtmass['jerr'][:len(gdt)] = tmass['e_jmag'][gdt]
    vtmass['hmag'][:len(gdt)] = tmass['hmag'][gdt]
    vtmass['herr'][:len(gdt)] = tmass['e_hmag'][gdt]
    vtmass['kmag'][:len(gdt)] = tmass['kmag'][gdt]
    vtmass['kerr'][:len(gdt)] = tmass['e_kmag'][gdt]
    #for c in virac.colnames:
    for c in vtmass.dtype.names:
        vtmass[c][len(gdt):] = virac[c][gdv]
    vtmass = Table(vtmass)
    #vtmass['ksmag'].name = 'kmag'
    #vtmass['kserr'].name = 'kerr'

    # 2MASS flag
    vtmass['is2mass'] = False
    vtmass['is2mass'][:len(gdt)] = True
    
    #plt.figure(5)
    #o=pl.hist2d(vtmass['jmag']-vtmass['kmag'],vtmass['hmag'],xr=[-2,8],yr=[19,4],log=True,
    #            xtitle='J-Ks',ytitle='H',title='VVV+2MASS')

    #from the Smith+2025 virac2 paper, the photometry is on the VISTA native system

    #virac2 has 545,346,537 unique stars
    
    return vtmass

def mergeglimpse(vtmass,glimpse,dcr=1.5):
    """ Merge with GLIMPSE and add RJCE dereddening columns """
    
    #glimpse = Table.read('glimpse_l340_b1.0_rad1.0.fits')
    ind1,ind2,dist = coords.xmatch(vtmass['ra'],vtmass['dec'],glimpse['ra'],glimpse['dec'],dcr,unique=True)
    print(len(ind1),'matches with GLIMPSE')
    
    #In [250]: len(vtmass),len(glimpse),len(ind1)
    #Out[250]: (3376478, 181885, 180483)
    #plt.figure(7)
    #o=pl.hist2d(vtmass['jmag'][ind1]-vtmass['kmag'][ind1],vtmass['hmag'][ind1],xr=[-2,8],yr=[19,4],log=True,
    #            xtitle='J-Ks',ytitle='H',title='VVV + Glimpse')
    #goes to H~15

    vtmass['glimpse_match'] = False
    vtmass['glimpse_match'][ind1] = True
    vtmass['glimpse_3_6mag'] = np.nan
    vtmass['glimpse_3_6mag'][ind1] = glimpse['_3_6mag'][ind2]
    vtmass['glimpse_4_5mag'] = np.nan
    vtmass['glimpse_4_5mag'][ind1] = glimpse['_4_5mag'][ind2]

    vtmass['ak'] = np.nan
    vtmass['ah'] = np.nan
    vtmass['jk0'] = np.nan
    vtmass['h0'] = np.nan
    gd, = np.where(np.isfinite(vtmass['hmag']) & np.isfinite(vtmass['glimpse_4_5mag']))
    vtmass['ak'][gd] = 0.918*(vtmass['hmag'][gd] - vtmass['glimpse_4_5mag'][gd] - 0.08)
    # ah = ak*1.55
    vtmass['h0'][gd] = vtmass['hmag'][gd]-vtmass['ak'][gd]*1.55
    # ejk = 1.50*ak
    gd, = np.where(np.isfinite(vtmass['hmag']) & np.isfinite(vtmass['glimpse_4_5mag']) &
                   np.isfinite(vtmass['jmag']) & np.isfinite(vtmass['kmag']))
    vtmass['jk0'][gd] = vtmass['jmag'][gd]-vtmass['kmag'][gd]-1.50*vtmass['ak'][gd]

    return vtmass
    
def selectrgb(vtmass,jkcut=None,hcut=None):
    
    # Select RGB stars
    if jkcut is None:
        jkcut = [0.90,1.4,1.4,0.90,0.90]
        hcut = [11.8+0.5,9.8,5.4,7.8,11.8+0.5]
    ind,cutind = dln.roi_cut(jkcut,hcut,vtmass['jk0'].data,vtmass['h0'].data)
    rgb = vtmass[cutind]
    print(len(cutind),'RGB stars')
    
    vtmass['rgb'] = False
    vtmass['rgb'][cutind] = True
    
    #plt.figure(8)
    #o=pl.hist2d(vtmass['jk0'],vtmass['h0'],xr=[-2,4],yr=[19,4],log=True,
    #            xtitle='(J-Ks)o',ytitle='Ho',title='Dereddened Photometry')
    #plt.plot(jkcut,hcut,c='r')

    #plt.figure(9)
    #o=pl.hist2d(rgb['jmag']-rgb['kmag'],rgb['hmag'],xr=[-2,8],yr=[19,4],log=True,
    #            xtitle='J-Ks',ytitle='H',title='RGB Candidates')

    #plt.figure(10)
    #o=pl.hist2d(vtmass['jmag'][ind1]-vtmass['kmag'][ind1],vtmass['hmag'][ind1],xr=[-2,8],yr=[19,4],log=True,
    #        xtitle='J-Ks',ytitle='H',title='VVV + Glimpse')
    #plt.scatter(rgb['jmag']-rgb['kmag'],rgb['hmag'],s=0.1,c='r')

    return rgb,vtmass

def mergegaia(vtmass,gaia,dcr=1.5):
    """ Merge VIRAC+2MASS with Gaia """

    #  Gaia
    ind1,ind2,dist = coords.xmatch(vtmass['ra'],vtmass['dec'],gaia['ra'],gaia['dec'],dcr,unique=True)
    #In [254]: len(vtmass),len(gaia),len(ind1)
    #Out[254]: (3376478, 1021936, 1001914)
    print(len(ind1),'matches with Gaia')

    # almost all gaia have matches
    #plt.figure(4)
    #o=pl.hist2d(vtmass['jmag'][ind1]-vtmass['kmag'][ind1],vtmass['hmag'][ind1],xr=[-2,8],yr=[19,4],log=True)
    #not great in the red

    vtmass['gaia_match'] = False
    vtmass['gaia_gmag'] = np.nan
    vtmass['gaia_gerr'] = np.nan
    vtmass['gaia_bpmag'] = np.nan
    vtmass['gaia_bperr'] = np.nan
    vtmass['gaia_rpmag'] = np.nan
    vtmass['gaia_rperr'] = np.nan
    vtmass['gaia_pmra'] = np.nan
    vtmass['gaia_pmra_error'] = np.nan
    vtmass['gaia_pmdec'] = np.nan
    vtmass['gaia_pmdec_error'] = np.nan
    #vtmass['gaia_parallax'] = np.nan
    #vtmass['gaia_parallax_error'] = np.nan

    vtmass['gaia_match'][ind1] = True
    vtmass['gaia_gmag'][ind1] = gaia['gmag'][ind2]
    vtmass['gaia_gerr'][ind1] = 1.0857*gaia['e_fg'][ind2]/gaia['fg'][ind2]
    vtmass['gaia_bpmag'][ind1] = gaia['bp'][ind2]
    vtmass['gaia_bperr'][ind1] = 1.0857*gaia['e_fbp'][ind2]/gaia['fbp'][ind2]
    vtmass['gaia_rpmag'][ind1] = gaia['rp'][ind2]
    vtmass['gaia_rperr'][ind1] = 1.0857*gaia['e_frp'][ind2]/gaia['frp'][ind2]
    vtmass['gaia_pmra'][ind1] = gaia['pmra'][ind2]
    vtmass['gaia_pmra_error'][ind1] = gaia['pmra_error'][ind2]
    vtmass['gaia_pmdec'][ind1] = gaia['pmdec'][ind2]
    vtmass['gaia_pmdec_error'][ind1] = gaia['pmdec_error'][ind2]
    #vtmass['gaia_parallax'][ind1] = gaia[''][ind2]
    #vtmass['gaia_parallax_error'][ind1] = gaia[''][ind2]
    
    # Check if we can use Gaia XP to pre-select metal-poor stars in the low-extinction region at higher b

    return vtmass

def mergedecaps(vtmass,decaps,dcr=1.5):
    """ Merge VIRAC+2MASS with DECaPS """

    
    ind1,ind2,dist = coords.xmatch(vtmass['ra'],vtmass['dec'],decaps['ra'],decaps['dec'],dcr,unique=True)
    #In [254]: len(vtmass),len(decaps),len(ind1)
    #(3376478, 6361425, 3053044)
    #almost all of the virac2+2mass sources have decaps matches
    print(len(ind1),'matches with decaps')
    
    #plt.figure(5)
    #o=pl.hist2d(vtmass['jmag'][ind1]-vtmass['kmag'][ind1],vtmass['hmag'][ind1],xr=[-2,8],yr=[19,4],log=True)

    #o=pl.hist2d(decaps['zmag'][ind2]-decaps['ymag'][ind2],decaps['ymag'][ind2],xr=[-1,3],yr=[23,12],log=True)

    vtmass['decaps_match'] = False
    vtmass['decaps_ndet'] = int
    vtmass['decaps_ndetg'] = int
    vtmass['decaps_gmag'] = np.nan
    vtmass['decaps_gerr'] = np.nan
    vtmass['decaps_ndetr'] = int
    vtmass['decaps_rmag'] = np.nan
    vtmass['decaps_rerr'] = np.nan
    vtmass['decaps_ndeti'] = int
    vtmass['decaps_imag'] = np.nan
    vtmass['decaps_ierr'] = np.nan
    vtmass['decaps_ndetz'] = int
    vtmass['decaps_zmag'] = np.nan
    vtmass['decaps_zerr'] = np.nan
    vtmass['decaps_ndety'] = int
    vtmass['decaps_ymag'] = np.nan
    vtmass['decaps_yerr'] = np.nan

    vtmass['decaps_match'][ind1] = True
    vtmass['decaps_ndet'][ind1] = decaps['ndet'][ind2]
    vtmass['decaps_ndetg'][ind1] = decaps['ndetg'][ind2]
    vtmass['decaps_gmag'][ind1] = decaps['gmag'][ind2]
    vtmass['decaps_gerr'][ind1] = decaps['gmagerr'][ind2]
    vtmass['decaps_ndetr'][ind1] = decaps['ndetr'][ind2]
    vtmass['decaps_rmag'][ind1] = decaps['rmag'][ind2]
    vtmass['decaps_rerr'][ind1] = decaps['rmagerr'][ind2]
    vtmass['decaps_ndeti'][ind1] = decaps['ndeti'][ind2]
    vtmass['decaps_imag'][ind1] = decaps['imag'][ind2]
    vtmass['decaps_ierr'][ind1] = decaps['imagerr'][ind2]
    vtmass['decaps_ndetz'][ind1] = decaps['ndetz'][ind2]
    vtmass['decaps_zmag'][ind1] = decaps['zmag'][ind2]
    vtmass['decaps_zerr'][ind1] = decaps['zmagerr'][ind2]
    vtmass['decaps_ndety'][ind1] = decaps['ndety'][ind2]
    vtmass['decaps_ymag'][ind1] = decaps['ymag'][ind2]
    vtmass['decaps_yerr'][ind1] = decaps['ymagerr'][ind2]

    return vtmass

def plots(vtmass,glon,glat):
    """ Make plots """
    
    backend = matplotlib.rcParams['backend']
    matplotlib.use('Agg')

    fig = plt.figure(figsize=(10,10))
    
    # All sources
    o=pl.hist2d(vtmass['jmag']-vtmass['kmag'],vtmass['hmag'],xr=[-2,8],yr=[19,4],log=True,
                xtitle='J-Ks',ytitle='H',title='All Sources ('+str(len(vtmass))+')')
    outfile = 'plots/all_cmd_l{:.0f}_b{:.1f}.png'.format(glon,glat)
    plt.savefig(outfile,bbox_inches='tight')

    # GLIMPSE matches
    ind, = np.where(vtmass['glimpse_match'])
    o=pl.hist2d(vtmass['jmag'][ind]-vtmass['kmag'][ind],vtmass['hmag'][ind],xr=[-2,8],yr=[19,4],log=True,
                xtitle='J-Ks',ytitle='H',title='GLIMPSE matches ('+str(len(ind))+')')
    outfile = 'plots/glimpsematches_cmd_l{:.0f}_b{:.1f}.png'.format(glon,glat)
    plt.savefig(outfile,bbox_inches='tight')
    
    # Gaia matches
    ind, = np.where(vtmass['gaia_match'])
    o=pl.hist2d(vtmass['jmag'][ind]-vtmass['kmag'][ind],vtmass['hmag'][ind],xr=[-2,8],yr=[19,4],log=True,
                xtitle='J-Ks',ytitle='H',title='Gaia matches ('+str(len(ind))+')')
    outfile = 'plots/gaiamatches_cmd_l{:.0f}_b{:.1f}.png'.format(glon,glat)
    plt.savefig(outfile,bbox_inches='tight')

    # DECaPS matches
    ind, = np.where(vtmass['decaps_match'])
    o=pl.hist2d(vtmass['jmag'][ind]-vtmass['kmag'][ind],vtmass['hmag'][ind],xr=[-2,8],yr=[19,4],log=True,
                xtitle='J-Ks',ytitle='H',title='DECaPS matches ('+str(len(ind))+')')
    outfile = 'plots/decapsmatches_cmd_l{:.0f}_b{:.1f}.png'.format(glon,glat)
    plt.savefig(outfile,bbox_inches='tight')

    # Dereddened CMD
    ind, = np.where(vtmass['glimpse_match'])
    o=pl.hist2d(vtmass['jk0'][ind],vtmass['h0'][ind],xr=[-1.5,2.5],yr=[16,4],log=True,
                xtitle='(J-Ks)o',ytitle='Ho',title='Dereddened CMD ('+str(len(ind))+')')
    outfile = 'plots/deredcmd_cmd_l{:.0f}_b{:.1f}.png'.format(glon,glat)
    plt.savefig(outfile,bbox_inches='tight')
    
    # RGB candidates
    ind, = np.where(vtmass['rgb'])
    o=pl.hist2d(vtmass['jmag'][ind]-vtmass['kmag'][ind],vtmass['hmag'][ind],xr=[-2,8],yr=[19,4],log=True,
                xtitle='J-Ks',ytitle='H',title='RGB Candidates ('+str(len(ind))+')')
    outfile = 'plots/rgb_cmd_l{:.0f}_b{:.1f}.png'.format(glon,glat)
    plt.savefig(outfile,bbox_inches='tight')

    # RGB candidates, dereddened CMD
    ind, = np.where(vtmass['glimpse_match'])
    rgbind, = np.where(vtmass['rgb'])
    o=pl.hist2d(vtmass['jk0'][ind],vtmass['h0'][ind],xr=[-1.5,2.5],yr=[16,4],log=True,
                xtitle='(J-Ks)o',ytitle='Ho',title='RGB Candidates - Dereddened CMD')
    plt.scatter(vtmass['jk0'][rgbind],vtmass['h0'][rgbind],s=1,c='r')
    outfile = 'plots/rgbderedcmd_cmd_l{:.0f}_b{:.1f}.png'.format(glon,glat)
    plt.savefig(outfile,bbox_inches='tight')

    # RGB candidates, raw CMD
    rgbind, = np.where(vtmass['rgb'])
    o=pl.hist2d(vtmass['jmag']-vtmass['kmag'],vtmass['hmag'],xr=[-2,8],yr=[19,4],log=True,
                xtitle='(J-Ks)o',ytitle='Ho',title='RGB Candidates - Raw CMD')
    plt.scatter(vtmass['jmag'][rgbind]-vtmass['kmag'][rgbind],vtmass['hmag'][rgbind],s=1,c='r')
    outfile = 'plots/rgbcmd_cmd_l{:.0f}_b{:.1f}.png'.format(glon,glat)
    plt.savefig(outfile,bbox_inches='tight')
    
    matplotlib.use(backend)

def run(glon,glat):
    """ Merge all of the catalogs """
    
    virac = Table.read('virac2_l{:.0f}_b{:.1f}_rad1.0.fits'.format(glon,glat))
    virac = virac.filled()
    print(len(virac),'VIRAC2 sources')
    
    # Photometrically calibrate VIRAC2 to 2MASS
    virac = calibratevirac(virac)
    
    # Merge with 2MASS
    tmass = Table.read('tmass_l{:.0f}_b{:.1f}_rad1.0.fits'.format(glon,glat))
    tmass = tmass.filled()
    vtmass = mergetmass(virac,tmass)

    # Adding galactic coordinates
    coo = SkyCoord(vtmass['ra'],vtmass['dec'],unit='degree',frame='icrs')
    vtmass['glon'] = coo.galactic.l.degree
    vtmass['glat'] = coo.galactic.b.degree
    
    # Merge with GLIMPSE and do RJCE dereddening
    glimpse = Table.read('glimpse_l{:.0f}_b{:.1f}_rad1.0.fits'.format(glon,glat))
    glimpse = glimpse.filled()
    vtmass = mergeglimpse(vtmass,glimpse)
    
    # Merge with Gaia
    gaia = Table.read('gaiaedr3_l{:.0f}_b{:.1f}_rad1.0.fits'.format(glon,glat))
    gaia = gaia.filled()
    vtmass = mergegaia(vtmass,gaia)
    
    # Merge with DECaPS
    decaps = Table.read('decaps_l{:.0f}_b{:.1f}_rad1.0.fits'.format(glon,glat))
    decaps = decaps.filled()
    vtmass = mergedecaps(vtmass,decaps)

    # Select RGB stars
    rgb,vtmass = selectrgb(vtmass)

    # Make the plots
    plots(vtmass,glon,glat)
    
    return vtmass
