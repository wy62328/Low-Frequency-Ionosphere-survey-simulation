#~
import h5py
import numpy as np
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
import astropy.units as u
import healpy as hp
from astropy.utils.iers import conf
from tqdm import tqdm

conf.auto_download = False

def coord_trans(time):
    x = np.arange(0,12*512**2,1)
    theta,phi = hp.pixelfunc.pix2ang(512,x)
    crab = SkyCoord(phi*u.radian,
                    (np.pi/2-theta)*u.radian,
                    frame='galactic',
                    obstime = Time('2022-01-02T09:00:00.5', format='isot', scale='utc'))
    t = Time(time, format='isot', scale='utc')#'2022-01-02T09:00:00.5'
    location = EarthLocation(lon=-17.89 * u.deg,
                             lat=40 * u.deg,
                             height=2200 * u.m,)#.get_itrs(times)#地址 
    frame = AltAz( location=location,obstime=t)#地平坐标系
    crab_altaz = crab.transform_to(frame)
    return crab_altaz.alt.radian,crab_altaz.az.radian

def run_ts(dtime,filepath='/public/home/wangyue/workspace/Ionosphere_sim/Crime data process/Time_Series_0102_test.hdf5'):
    dtime = 30#min
    alt_collections = []
    az_collections = []
    for i in tqdm(range(1,int(24*60/dtime))):
        t = i*30
        hour,minute = int((t-t%60)/60),t%60
        alt,az = coord_trans('2022-01-02T{0}:{1}:00'.format(hour,minute))#银道坐标系到地平坐标系
        alt_collections.append(alt)
        az_collections.append(az)
    hf = h5py.File(filepath, 'w')
    hf.create_dataset('alt', data=alt_collections)
    hf.create_dataset('az', data=az_collections)
    hf.close()