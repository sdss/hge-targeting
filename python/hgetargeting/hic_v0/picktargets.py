import os
import numpy as np
from astropy.table import Table,vstack
from dlnpyutils import utils as dln,coords
from astropy.io import fits
import traceback
import healpy as hp
from astropy.coordinates import SkyCoord
import matplotlib
import matplotlib.pyplot as plt

def picktargets(hpix):
    """ pick targets for one healpix """

    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/hic/ring256/'
    filename = basedir+'hic_{:d}.fits'.format(hpix)
    tab = Table.read(filename)
    tab = tab.filled()
    print(len(tab),'sources')
    
    gd, = np.where(np.isfinite(tab['jk0']) & np.isfinite(tab['h0']) & np.isfinite(tab['hmag']))
    if len(gd)==0:
        return [],[]
    tab_good = tab[gd]
    
    # RGB
    # ---- RGB selection ----
    jkcut = [0.90,1.4,1.4,0.90,0.90]
    hcut = [11.8+1.0,9.8+1.0,5.4-1,7.8-1,11.8+1.0]
    ind,cutind = dln.roi_cut(jkcut,hcut,tab_good['jk0'],tab_good['h0'])
    rgb = tab_good[cutind]
    gdhmag, = np.where((rgb['hmag']>=11) & (rgb['hmag']<=15.5))
    rgb = rgb[gdhmag]
    print(len(rgb),'RGB targets')
    
    
    # RC
    # ---- RC selection ----
    jkcut_rc = [0.55,0.90,0.90,0.55,0.55]
    hcut_rc = [14.0,14.0,10.0,10.0,14.0]
    ind,cutind_rc = dln.roi_cut(jkcut_rc,hcut_rc,tab_good['jk0'],tab_good['h0'])
    rc = tab_good[cutind_rc]
    gdhmag, = np.where((rc['hmag']>=14) & (rc['hmag']<=15.5))
    rc = rc[gdhmag]
    print(len(rc),'RC targets')
    
    # ---- Close neighbors ----
    # fibers are 1.3" in diameter
    # use Gaia and nir as master catalogs
    #brt_nir,fcontam_nir = brightneighbors(nir,alltargets,'hmag')
    #alltargets['brtnei'] = brt_nir
    #alltargets['brtnei_fcontam'] = fcontam_nir
    #print('{:d} targets with close neighbors ({:.2f}%)'.format(np.sum(brt_nir),np.sum(brt_nir)/len(alltargets)*100))
    
    return rgb,rc

def pickall(clobber=False):
    """ pick targets """

    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/hic/'
    targsdir = basedir+'targets/'
    
    allpix = dln.readlines('allpix.txt')
    allpix = np.array(allpix).astype(int)

    for i in range(len(allpix)):
        outfile = targsdir+'targets_{:d}.fits'.format(allpix[i])
        if os.path.exists(outfile)==True and clobber==False:
            print(outfile,'exists')
            continue
        try:
            rgb,rc = picktargets(allpix[i])
            hdulist = fits.HDUList()
            hdulist.append(fits.table_to_hdu(rgb))
            hdulist.append(fits.table_to_hdu(rc))
            hdulist[0].header['nrgb'] = len(rgb)
            hdulist[0].header['nrgb11'] = np.sum((rgb['hmag']>=11) & (rgb['hmag']<=12))
            hdulist[0].header['nrgb12'] = np.sum((rgb['hmag']>=12) & (rgb['hmag']<=13))
            hdulist[0].header['nrgb13'] = np.sum((rgb['hmag']>=13) & (rgb['hmag']<=14))
            hdulist[0].header['nrgb14'] = np.sum((rgb['hmag']>=14) & (rgb['hmag']<=15.5))
            hdulist[0].header['nrc'] = len(rc)
            hdulist[0].header['nrc11'] = np.sum((rc['hmag']>=11) & (rc['hmag']<=12))
            hdulist[0].header['nrc12'] = np.sum((rc['hmag']>=12) & (rc['hmag']<=13))
            hdulist[0].header['nrc13'] = np.sum((rc['hmag']>=13) & (rc['hmag']<=14))
            hdulist[0].header['nrc14'] = np.sum((rc['hmag']>=14) & (rc['hmag']<=15.5))
            hdulist.writeto(outfile,overwrite=True)
            
            print(i+1,allpix[i],len(rgb),len(rc))
            print('')
            
        except KeyboardInterrupt:
            raise
        except:
            traceback.print_exc()
            
        #import pdb; pdb.set_trace()

def summary():

    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/hic/'
    targsdir = basedir+'targets/'
    
    allpix = dln.readlines('allpix.txt')
    allpix = np.array(allpix).astype(int)

    dt = [('hpix',int),('ra',float),('dec',float),('l',float),('b',float),
          ('nrgb',int),('nrgb11',int),('nrgb12',int),('nrgb13',int),('nrgb14',int),
          ('nrc',int)]
    sumtab = np.zeros(len(allpix),dtype=np.dtype(dt))
    sumtab['hpix'] = allpix
    ra,dec = hp.pix2ang(256,allpix,lonlat=True)
    sumtab['ra'] = ra
    sumtab['dec'] = dec
    coo = SkyCoord(ra,dec,unit='degree',frame='icrs')
    sumtab['l'] = coo.galactic.l.degree
    sumtab['b'] = coo.galactic.b.degree
    for i in range(len(allpix)):
        outfile = targsdir+'targets_{:d}.fits'.format(allpix[i])
        if os.path.exists(outfile):
            hdu = fits.open(outfile)
            sumtab['nrgb'][i] = hdu[0].header['nrgb']
            sumtab['nrgb11'][i] = hdu[0].header['nrgb11']
            sumtab['nrgb12'][i] = hdu[0].header['nrgb12']
            sumtab['nrgb13'][i] = hdu[0].header['nrgb13']
            sumtab['nrgb14'][i] = hdu[0].header['nrgb14']
            sumtab['nrc'][i] = hdu[0].header['nrc']
            hdu.close()

        print(i+1,allpix[i],sumtab['nrgb'][i],sumtab['nrc'][i])
        
    Table(sumtab).write('hge_targets_summary.fits',overwrite=True)

def rccolor():
    """ Calculate RC color as a check of extinction. """

    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/hic/ring256/'    
    allpix = dln.readlines('allpix.txt')
    allpix = np.array(allpix).astype(int)

    outfile = 'rccolor.fits'
    
    dt = [('hpix',int),('ntotal',int),('nrcpeak',int),('jkpeak',float),('jkwt',float),('jksig',float)]
    sumtab = np.zeros(len(allpix),dtype=np.dtype(dt))
    sumtab = Table(sumtab)
    sumtab['hpix'] = allpix
    sumtab['ntotal'] = -1
    sumtab['nrcpeak'] = -1
    sumtab['jkpeak'] = np.nan
    sumtab['jkwt'] = np.nan
    sumtab['jksig'] = np.nan
    
    for i in range(len(allpix)):
        filename = basedir+'hic_{:d}.fits'.format(sumtab['hpix'][i])
        if os.path.exists(filename)==False:
            print(filename,'not found')
            continue
        tab = Table.read(filename)
        tab = tab.filled()
        #print(len(tab),'sources')
        sumtab['ntotal'][i] = len(tab)
        
        gd, = np.where(np.isfinite(tab['jk0']) & np.isfinite(tab['h0']) & np.isfinite(tab['hmag']))
        if len(gd)==0:
            continue
        tab2 = tab[gd]
    
    
        # RC
        # ---- RC selection ----
        #jkcut_rc = [0.55,0.90,0.90,0.55,0.55]
        #hcut_rc = [14.0,14.0,10.0,10.0,14.0]
        #ind,cutind_rc = dln.roi_cut(jkcut_rc,hcut_rc,tab_good['jk0'],tab_good['h0'])
        #rc = tab_good[cutind_rc]
        #gdhmag, = np.where((rc['hmag']>=14) & (rc['hmag']<=15.5))
        #rc = rc[gdhmag]
        #print(len(rc),'RC targets')

        h0range = [12.0,13.0]
        #h0range = [11.5,12.5]
        ind, = np.where((tab2['jk0']>0.4) & (tab2['jk0']<1.1) &
                        (tab2['h0']>h0range[0]) & (tab2['h0']<h0range[1]))
        jkbins = np.arange(0.4,1.2,0.1)
        hist1,xhist1 = np.histogram(tab2['jk0'][ind],bins=jkbins)
        xhist1 = xhist1[:-1]+(xhist1[1]-xhist1[0])*0.5
        bestind1 = np.argmax(hist1)
        jkpeak1 = xhist1[bestind1]
        jkbins2 = np.arange(jkpeak1-0.2,jkpeak1+0.2,0.05)
        hist2,xhist2 = np.histogram(tab2['jk0'][ind],bins=jkbins2)
        xhist2 = xhist2[:-1]+(xhist2[1]-xhist2[0])*0.5
        bestind2 = np.argmax(hist2)
        jkpeak2 = xhist2[bestind2]
        jkwt = np.sum(hist2*xhist2)/np.sum(hist2)
        jksig = np.sqrt(np.sum(hist2*(xhist2-jkwt)**2)/np.sum(hist2))

        sumtab['nrcpeak'][i] = hist2[bestind2]
        sumtab['jkpeak'][i] = jkpeak2
        sumtab['jkwt'][i] = jkwt
        sumtab['jksig'][i] = jksig

        print('{:d} {:d} {:d} {:.3f} {:.3f} {:.3f}'.format(i+1,sumtab['hpix'][i],sumtab['nrcpeak'][i],
                                                           sumtab['jkpeak'][i],sumtab['jkwt'][i],sumtab['jksig'][i]))

        if i % 100 == 0 and i>0:
            sumtab.write(outfile,overwrite=True)

        #import pdb; pdb.set_trace()
        
    sumtab.write(outfile,overwrite=True)

    import pdb; pdb.set_trace()

def pickfieldtargets():
    """ Pick targets for a field """

    basedir = '/uufs/chpc.utah.edu/common/home/sdss42/sdsswork/users/u0914350/as5/hge/hic/'
    targsdir = basedir+'targets/'
    
    fields = Table.read('../hge_fields_091526.fits')
    fields['nrgb'] = 0
    fields['nrgb11'] = 0
    fields['nrgb12'] = 0
    fields['nrgb13'] = 0
    fields['nrgb14'] = 0
    fields['nrc'] = 0
    for i in range(len(fields)):
        pix = hp.ang2pix(256,fields['racen'][i],fields['deccen'][i],lonlat=True)
        vec = hp.pix2vec(256,pix)
        querypix = hp.query_disc(256,vec,np.deg2rad(fields['radius'][i]*1.2),inclusive=True)
        print(i+1,fields['racen'][i],fields['deccen'][i],len(querypix))
        rgbtargs,rctargs = [],[]
        for j in range(len(querypix)):
            filename = targsdir+'targets_{:d}.fits'.format(querypix[j])
            if os.path.exists(filename):
                rgbtargs1 = Table.read(filename,1)
                rctargs1 = Table.read(filename,2)
                if len(rgbtargs1):
                    rgbtargs.append(rgbtargs1)
                    rctargs.append(rctargs1)
            else:
                print(filename,'not found')
        rgbtargs = vstack(rgbtargs)
        rctargs = vstack(rctargs)
        # Make radius cut
        rgbdist = coords.sphdist(fields['racen'][i],fields['deccen'][i],rgbtargs['ra'],rgbtargs['dec'])
        gdrgb, = np.where(rgbdist <= fields['radius'][i])
        rgbtargs = rgbtargs[gdrgb]
        rcdist = coords.sphdist(fields['racen'][i],fields['deccen'][i],rctargs['ra'],rctargs['dec'])
        gdrc, = np.where(rcdist <= fields['radius'][i])
        rctargs = rctargs[gdrc]

        fields['nrgb'][i] = len(rgbtargs)
        fields['nrgb11'][i] = np.sum((rgbtargs['hmag']>=11) & (rgbtargs['hmag']<12))
        fields['nrgb12'][i] = np.sum((rgbtargs['hmag']>=12) & (rgbtargs['hmag']<13))
        fields['nrgb13'][i] = np.sum((rgbtargs['hmag']>=13) & (rgbtargs['hmag']<14))
        fields['nrgb14'][i] = np.sum((rgbtargs['hmag']>=14) & (rgbtargs['hmag']<15.5))
        fields['nrc'][i] = len(rctargs)
        print('  ',len(rgbtargs),len(rctargs))
        
        # Save the targets
        outfile = basedir+'fieldtargets/fieldtargets_{:d}.fits'.format(fields['fieldid'][i])
        hdulist = fits.HDUList()
        hdulist.append(fits.table_to_hdu(rgbtargs))
        hdulist.append(fits.table_to_hdu(rctargs))
        hdulist[0].header['obs'] = fields['observatory'][i]
        hdulist[0].header['racen'] = fields['racen'][i]
        hdulist[0].header['deccen'] = fields['deccen'][i]
        hdulist[0].header['radius'] = fields['radius'][i]
        hdulist.writeto(outfile,overwrite=True)

        #import pdb; pdb.set_trace()

    fields.write(basedir+'fieldtargets/fieldtargets_summary.fits')

    import pdb; pdb.set_trace()

def fieldtargetplots():
    """ Make plots of field targets """

    tab = Table.read('fieldtargets_summary.fits')
    tab['glon'].name = 'l'
    tab['glat'].name = 'b'

    cols = ['nrgb','nrgb11','nrgb12','nrgb13','nrgb14','nrc']
    labels = ['RGB','RGB (11<H<12)','RGB (12<H<13)','RGB (13<H<14)','RGB (14<H<15.5)','RC']

    for i in range(len(cols)):
        col = cols[i]
        label = labels[i]
        
        fig,ax = plt.subplots(figsize=(24,4))
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.12, top=0.96)

        values = np.asarray(tab[col])
        norm = matplotlib.colors.LogNorm(
            vmin=values[values > 0].min(),
            vmax=values.max()
        )
        #norm = matplotlib.colors.LogNorm(vmin=25, vmax=max(26, values.max()))

        
        #norm = matplotlib.colors.Normalize(vmin=np.min(tab[col]),vmax=np.max(tab[col]))    
        cmap = plt.get_cmap("plasma")
        phi = np.linspace(0,2*np.pi,100)
        for i in range(len(tab)):
            #ax.plot(tab['radius'][i]*np.sin(phi)+tab['l'][i],
            #        tab['radius'][i]*np.cos(phi)+tab['b'][i],c='k',linewidth=1)
            #color = cmap(norm(values[i])) if values[i] > 0 else "lightgray"
            #ax.fill(tab['radius'][i]*np.sin(phi)+tab['l'][i],
            #        tab['radius'][i]*np.cos(phi)+tab['b'][i],color=color)

            x = tab['radius'][i]*np.sin(phi) + tab['l'][i]
            y = tab['radius'][i]*np.cos(phi) + tab['b'][i]
            n = values[i]

            #ax.fill(x, y,
            #        facecolor=cmap(norm(n)) if n > 0 else 'lightgray',
            #        edgecolor='k',
            #        linewidth=0.5,
            #        hatch='///' if 0 < n < 25 else None)

            if n == 0:
                ax.fill(x, y, facecolor='lightgray', edgecolor='k', linewidth=0.5)
            elif n < 25:
                #with matplotlib.rc_context({'hatch.linewidth': 2.0}):
                #    ax.fill(
                #        x, y,
                #        facecolor='mistyrose',
                #        edgecolor='crimson',  # sets hatch and outline color
                #        linewidth=1.2,
                #        hatch='///'
                #    )
                with matplotlib.rc_context({'hatch.linewidth': 1.5}):
                    ax.fill(
                        x, y,
                        facecolor=cmap(norm(n)) if n > 0 else 'lightgray',
                        edgecolor='0.65',  # light-gray hatching
                        linewidth=0,
                        hatch='xxxx',
                        alpha=0.8
                    )
                # Separate black outline
                ax.plot(x, y, color='k', linewidth=0.5)
                    
            else:
                ax.fill(x, y, facecolor=cmap(norm(n)),
                        edgecolor='k', linewidth=0.5)



                
            
        plt.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                     ax=ax,label='Number of {:s} Targets'.format(label))

        plt.xlim(63,-63)
        plt.ylim(-8,8)
        plt.xlabel('Galactic Longitude (deg)')
        plt.ylabel('Galactic Latitue (deg)')
        plt.title('Number of {:s} Targets per HGE Field'.format(label))

        plt.savefig('hge_fieldtargets_{:s}.png'.format(col),bbox_inches='tight')
        
    import pdb; pdb.set_trace()
