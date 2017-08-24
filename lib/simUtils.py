description = """a module that holds helper functions for the injection simulations and subsequent skymap analysis"""
authors = "Reed Essick (reed.essick@ligo.org), Hsin-Yu Chen (hsinyuchen@uchicago.edu)"

#-------------------------------------------------

import numpy as np
import healpy as hp

import astropy
from astropy.time import Time as astropyTime
from astropy import coordinates as astropyCoordinates

import ephem

from lal.gpstime import tconvert
from lal.lal import GreenwichMeanSiderealTime as GMST

#-------------------------------------------------

deg2rad = np.pi/180
rad2deg = 180/np.pi

twopi = 2*np.pi
pi2 = np.pi/2
pi6 = np.pi/6
pi8 = np.pi/8
pi10 = np.pi/10

hour = 3600.
day = 86400.

t_ephem_start = astropyTime('1899-12-31 12:00:00', format='iso', scale='utc')

#-------------------------------------------------

### known observatories

class Observatory(object):

    def __init__(self, name, lon, lat, color='r', marker='o', degrees=True):
        """
        if degrees, we expect lon and lat to be provided in degrees and we convert them to radians internally
        """
        self.name = name
        self.lon = lon
        self.lat = lat
        if degrees:
            self.lon *= deg2rad
            self.lat *= deg2rad
        self.color = color
        self.marker = marker

### set up dictionary of known observatories
observatories = {}
observatories['Palomar'] = Observatory( 'Palomar', -116.8639, 33.3558, color='r', marker='o', degrees=True )
observatories['Cerro Tololo'] = Observatory( 'Cerro Tololo', -70.806525, -30.169661, color='m', marker='h', degrees=True )
observatories['Mauna Kea'] = Observatory( 'Mauna Kea', -156.25333, 20.70972, color='y', marker='D', degrees=True )
observatories['La Palma'] = Observatory( 'La Palma', -17.8947, 28.7636, color='b', marker='s', degrees=True )
observatories['Anglo-Australian'] = Observatory( 'Anglo-Australian', 149.0672, -31.2754, color='g', marker='^', degrees=True )
observatories['Nishi-Harima'] = Observatory( 'Nishi-Harima', 134.3356, 35.0253, color='c', marker='p', degrees=True )

#-------------------------------------------------

### utilities for drawing random times

def gps2relativeUTC( t ):
    """
    converts gps secons into the number of seconds after the most recent 00:00:00 UTC
    """
    if isinstance(t, (np.ndarray, list, tuple)):
        times = []
        for T in t:
            ans = tconvert( T )
            times.append( T - float(tconvert( ans.split(',')[0]+" 00:00:00" )) )
        return np.array(times)
    else:
        ans = tconvert( t )
        return t-float(tconvert( ans.split(',')[0]+" 00:00:00" ))

def diurnalPDF( t, amplitude=0.5, phase=pi8 ):
    """
    "t" must be specified in gps seconds
    we convert the time in gps seconds into the number of seconds after the most recent 00:00:00 UTC
    return (1 + amplitude*sin(2*pi*t/day - phase))/day
    """
    if amplitude > 1:
        raise ValueError("amplitude cannot be larger than 1")
    t = gps2relativeUTC(t)
    return (1 + amplitude*np.sin(twopi*t/day - phase))/day

def diurnalCDF( t, amplitude=0.5, phase=pi8 ):
    """
    "t" must be specified in gps seconds
    we convert the time in gps seconds into the number of seconds after the most recent 00:00:00 UTC
    return t/day - (amplitude/2pi)*cos(2*pi*t/day - phase)
    """
    if amplitude > 1:
        raise ValueError("amplitude cannot be larger than 1")
    t = gps2relativeUTC(t)
    return t/day - (amplitude/twopi)*(np.cos(twopi*t/day - phase) - np.cos(phase))

def drawTimes( start, end, N, amplitude=0.5, phase=pi8, verbose=False ):
    """
    draws N times from the diurnal cycle between start and end
    """
    if start >= end:
        raise ValueError("bad start and end times")

    dur = end-start
    times = []
    maxPDF = (1+amplitude)/day
    while len(times) < N:
        t, p = np.random.rand(2)
        t = start + t*dur
        if diurnalPDF( t, amplitude=amplitude, phase=phase ) > p*maxPDF: ### accept the time if the CDF is bigger than the random draw
            if verbose:
                print "accepted t=%.9f"%(t)
            times.append( t )

    return times

#-------------------------------------------------

### utilities for rotating maps

def rotateRAC2C( ra, gps1, gps2 ):
    """
    rotates the RA according to the change in gps

    takes ra at gps1 and rotates it so that the earth-fixed coordinates are invarient but the time has changed to gps2
    """
    gmst2 = GMST( gps2 )
    gmst1 = GMST( gps1 )

    return (ra - gmst1 + gmst2)%(twopi)

def rotateRAC2E( ra, gps ):
    """
    rotates ra -> earth fixed coords
    """
    gmst = GMST( gps )
    return (ra - gmst)%(twopi) 

def rotateRAE2C( phi, gps ):
    """
    rotates earth fixed coords -> ra
    """
    gmst = GMST( gps )
    return (phi + gmst)%(twopi)

def rotateMap( posterior, dphi ):
    """
    rotates phi -> phi+dphi
    """
    npix = len(posterior)
    nside = hp.npix2nside( npix )
    theta, phi = hp.pix2ang( nside, np.arange(npix) )
    phi += dphi

    new_pix = hp.ang2pix( nside, theta, phi )

    return posterior[new_pix]

def rotateMapC2C( posterior, old_gps, new_gps ):
    """
    returns a rotated map that keeps the posterior in the same relative position to the detectors at the new_gps time 
    as it was at the old_gps time.
    """
    npix = len(posterior)
    nside = hp.npix2nside( npix )

    theta, new_phi = hp.pix2ang( nside, np.arange( npix ) )
    phi = rotateRAC2C( new_phi, new_gps, old_gps ) ### rotate the RA according to times
                                                   ### need to map the RAs at the new gps time into the RAs at the old gps time

    new_pix = hp.ang2pix( nside, theta, phi )

    return posterior[new_pix]

def rotateMapC2E( posterior, gps ):
    npix = len(posterior)
    nside = hp.npix2nside( npix )

    theta, phi = hp.pix2ang( nside, np.arange( npix ) )
    ra = rotateRAE2C( phi, gps ) ### rotate phi to get ra -> needed to determine original indexing

    new_pix = hp.ang2pix( nside, theta, ra )

    return posterior[new_pix]

def rotateMapE2C( posterior, gps ):
    npix = len(posterior)
    nside = hp.npix2nside( npix )

    theta, ra = hp.pix2ang( nside, np.arange( npix ) )
    phi = rotateRAC2E( ra, gps ) ### rotate the RA to get phi -> needed to determine original indexing

    new_pix = hp.ang2pix( nside, theta, phi )

    return posterior[new_pix]

#-------------------------------------------------

def solarPosition( gps, coord="C" ):
    '''
    '''
    timeObj = astropyTime( tconvert(int(gps), form="%Y-%m-%dT%H:%M:%S")+("%.3f"%(gps%1))[1:], format='isot', scale='utc')
    sun = astropyCoordinates.get_sun(timeObj)
    if coord=="C":
        return float(sun.dec.radian), float(sun.ra.radian)
    else:
        return float(sun.dec.radian), rotateRAC2E( float(sun.ra.radian), gps )

def lunarPosition( gps, coord="C" ):
    '''
    '''
    moon = ephem.Moon()
    moon.compute(tconvert(int(gps), form="%Y/%m/%d %H:%M:%S"))
    if coord=="C":
        return float(moon.dec), float(moon.ra)
    else:
        return float(moon.dec), rotateRAC2E( float(moon.ra), gps )

### utilities for generating masks for occlusion

def solarOcclusion(sunDec, sunRA, nside, dead_zone, coord="C"):
    '''
    Find the accesible part of the sky in equatorial coordinate.

    "t" must be specified in gps seconds and the "dead_zone" must be specified in radians.
    returns a mask which corresponds to only those parts of the sky that are further than "dead_zone" from the sun at this time

    adapted from Hsin-Yu Chen's code
    '''
    if coord!="C":
        print "we only know how to handle coord=\"C\" at the moment. Returning a mask that removes nothing..."
        return np.ones((hp.nside2npix(nside),), dtype="int")

    npix = hp.nside2npix( nside )

    ### get solar position in spherical coordinate
    sunTheta = pi2 - sunDec

    ### get the cartesian coordinates of all the other pixels
    pix = np.arange( npix )
    theta, phi = hp.pix2ang( nside, pix )

    ### compute cos(theta) between all pixels and the sun in spherical coordinates
    cosdtheta = np.cos(sunTheta)*np.cos(theta) + np.sin(sunTheta)*np.sin(theta)*np.cos(sunRA-phi)

    return (cosdtheta <= np.cos(dead_zone)).astype(int)

def lunarOcclusion(lunDec, lunRA, nside, dead_zone, coord="C"):
    '''
    Find the accesible part of the sky in equatorial coordinate.

    "t" must be specified in gps seconds and the "dead_zone" must be specified in radians.
    returns a mask which corresponds to only those parts of the sky that are further than "dead_zone" from the sun at this time

    adapted from Hsin-Yu Chen's code
    '''
    if coord!="C":
        print "we only know how to handle coord=\"C\" at the moment. Returning a mask that removes nothing..."
        return np.ones((hp.nside2npix(nside),), dtype="int")

    npix = hp.nside2npix( nside )

    ### get solar position in spherical coordinate
    lunTheta = pi2 - lunDec

    ### get the cartesian coordinates of all the other pixels
    pix = np.arange( npix )
    theta, phi = hp.pix2ang( nside, pix )

    ### compute cos(theta) between all pixels and the sun in spherical coordinates
    cosdtheta = np.cos(lunTheta)*np.cos(theta) + np.sin(lunTheta)*np.sin(theta)*np.cos(lunRA-phi)

    return (cosdtheta <= np.cos(dead_zone)).astype(int)

#-------------------------------------------------

### utilities for determining where an observatory can see

def observableRegion( sunDec, sunRA, nside, obsAng, obsLat, obsLon, solar_occlusion_angle=pi6, coord='C' ):
    """
    computes the region that is just observable for a particular observatory
    assumes the sun does not move and computes bounds based on the horizon implied by obsAng and solar_occlusion_angle
    """
    ### figure out where the sun is
    if coord!="C":
        print "we only know how to handle coord=\"C\" at the moment. Returning a mask that removes nothing..."
        return np.ones((hp.nside2npix(nside),), dtype="int")

    ### get solar position in spherical coordinate
    theta_sun = pi2 - sunDec
    cosTheta_sun = np.cos(theta_sun)
    sinTheta_sun = np.sin(theta_sun)

    ### get the cartesian coordinates of all the other pixels
    npix = hp.nside2npix( nside )

    ### cut out regions of the sky that are not visible ever
    obsTheta = pi2-obsLat
    cosObsTheta = np.cos(obsTheta)
    sinObsTheta = np.sin(obsTheta)

    ### add back in the bit of sky the observatory can see when the sun has just set or risen
    ### need to find out where the "effective terminator" lives
    ### read off the associated longitudes for obsLat from the effective terminator
    ### add back in the bits of sky that are observable by the observatory (a circle around this point with radius=obsAng)

    ### compute the angular distance between phi_sun and the effective terminator
    cosdphi = (np.cos(solar_occlusion_angle+obsAng) - cosTheta_sun*cosObsTheta)/(sinTheta_sun*sinObsTheta)

    if cosdphi < -1: ### polar day -> never can observer
        return np.zeros((npix,), dtype=int) ### black out the entire sky

    elif cosdphi > 1: ### polar night -> always can observe
        theta, _ = hp.pix2ang( nside, np.arange(npix) )
        return (obsTheta-obsAng<=theta)*(theta<=obsTheta+obsAng) ### allow as much visibility as possible determined by obsAng

    else: ### there is a sunrise and a sunset  
        ### there will be times when this latitude can observe, so we figure out the change in RA associated therewith 
        dphi = np.arccos( cosdphi )

        ### compute a few helpful numbers
        theta, phi = hp.pix2ang( nside, np.arange(npix) )
        cosTheta = np.cos(theta)
        sinTheta = np.sin(theta)

        cosObsAng = np.cos(obsAng)
        A = cosTheta*cosObsTheta
        B = sinTheta*sinObsTheta

        ### add in the regions corresponding to the observatory's visible region when it sits at the effective terminator
        obsReg = ( cosObsAng <= A + B*np.cos( sunRA+dphi - phi ) ) + ( cosObsAng <= A + B*np.cos( sunRA-dphi - phi ) )

        ### allow only those parts of the sky that are far enough from the sun near twilight
        deltaPhi1 = phi-sunRA
        deltaPhi1[deltaPhi1<0] += twopi
        deltaPhi2 = sunRA-phi
        deltaPhi2[deltaPhi2<0] += twopi
        ###                basic requirement form latitude             requirements from longitude
        obsReg += (obsTheta-obsAng<=theta)*(theta<=obsTheta+obsAng) * ((deltaPhi1 > dphi )*(deltaPhi2 > dphi))
       
        ### return result as integer array
        return (obsReg>0).astype(int)

def unrealistic_observableRegion( nside, obsAng, obsLat ):
    """
    returns a mask for where the observatory could possibly see if occlusion was never an issue
    this is basically an annulus wrapped around the celestial sphere with width=2*obsAng
    """
    obsTheta = pi2-obsLat
    THETA = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))[0]
    return (obsTheta-obsAng<=THETA)*(THETA<=obsTheta+obsAng)

#-------------------------------------------------

### utilities for zenith distance

def zenithAng( obsLat, srcDec, srcRA, sunDec, sunRA, gps, obsAng, solarOcclusionAng=48*deg2rad ):
    """
    return the minimum distance an observatory at obsLat can observe a source at (srcDec, srcRA) at time gps
    all angle should be specified in radians
    """
    obsTheta = pi2-obsLat
    cosObsTheta = np.cos(obsTheta)
    sinObsTheta = np.sin(obsTheta)        

    sunTheta = pi2 - sunDec

    cosdphi = (np.cos(obsAng+solarOcclusionAng) - np.cos(sunTheta)*cosObsTheta)/(np.sin(sunTheta)*sinObsTheta)
    if cosdphi < -1: ### polar day
        return np.infty

    elif cosdphi > 1: ### polar night
        return np.abs(obsLat-srcDec)

    else: ### there is a sunrise and sunset    
        dphi = np.arccos(cosdphi)

        ### source is within daylight region -> minimum of distance at sunrise and sunset
        if np.abs(srcRA-sunRA)%twopi <= dphi:
            srcTheta = pi2 - srcDec
            cosSrcTheta = np.cos(srcTheta)
            sinSrcTheta = np.sin(srcTheta)

            A = cosSrcTheta*cosObsTheta 
            B = sinSrcTheta*sinObsTheta

            dangSunrise = np.arccos(A + B*np.cos(sunRA-dphi - srcRA))
            dangSunset  = np.arccos(A + B*np.cos(sunRA+dphi - srcRA))
            return min(dangSunrise, dangSunset)

        else: ### source is within night region -> difference in latitudes
            return np.abs(obsLat-srcDec)

def marge_zenithAng( post, obsLat, sunDec, sunRA, gps, obsAng, solarOcclusionAng=48*deg2rad ):
    """
    compute the zenith distance weighted by the probability from post.
    if mask is not provided, we set it to be 1 everywhere.

    returns Real -> probability that is observable
            Imag -> probability that is not observable
    """
    npix = len(post)
    nside = hp.npix2nside( npix )

    srcTheta, srcRA = hp.pix2ang( nside, np.arange(npix) )

    obsTheta = pi2-obsLat
    cosObsTheta = np.cos(obsTheta)
    sinObsTheta = np.sin(obsTheta)

    sunTheta = pi2 - sunDec

    cosdphi = (np.cos(obsAng+solarOcclusionAng) - np.cos(sunTheta)*cosObsTheta)/(np.sin(sunTheta)*sinObsTheta)
    if cosdphi < -1: ### polar day
        return np.infty

    elif cosdphi > 1: ### polar night
        dTheta = np.abs(obsTheta-srcTheta)
        isObs = dTheta <= obsAng
        isNotObs = (1-isObs).astype(bool)

        return np.sum( post[isObs]*dTheta[isObs] ) + 1.0j*np.sum( post[isNotObs]*dTheta[isNotObs] )

    else: ### there is a sunrise and a sunset
    
        ### fill in array
        zenAng = np.empty((npix,), dtype=complex)

        dphi = np.arccos(cosdphi)

        ### source is within daylight region -> minimum of distance at sunrise and sunset
        truth = np.abs(srcRA-sunRA)%twopi <= dphi

        cosSrcTheta = np.cos(srcTheta[truth])
        sinSrcTheta = np.sin(srcTheta[truth])

        A = cosSrcTheta*cosObsTheta
        B = sinSrcTheta*sinObsTheta

        dAngSunrise = np.arccos(A + B*np.cos(sunRA-dphi - srcRA[truth]))
        dAngSunset  = np.arccos(A + B*np.cos(sunRA+dphi - srcRA[truth]))

        ### check whether source is obser
        zenAng[truth] = np.min( np.array([dAngSunrise, dAngSunset]), axis=0 ) ### if it is within daylight band, take the minimum 

        ### for the rest, take the difference in latitudes
        truth = (1-truth).astype(bool)
        zenAng[truth] = np.abs(srcTheta[truth] - obsTheta)

        ### change all parts that are not observable to be imaginary
        ### closest angle is bigger than obsAng
        zenAng[zenAng>=obsAng] *= 1.0j

        return np.sum( post*zenAng )

#-------------------------------------------------

### utilities for determing when a location become observable

def dAngUntilSunrise( obsLat, obsLon, sunLat, sunLon, obsAng, solarOcclusionAng=48*deg2rad ):
    """
    returns hour angle in radians
    """
    obsTheta = pi2 - obsLat
    sunTheta = pi2 - sunLat
    A = np.cos(obsTheta)*np.cos(sunTheta)
    B = np.sin(obsTheta)*np.sin(sunTheta)

    dphi = np.arccos( (np.cos(obsAng+solarOcclusionAng) - A)/B )
    return (sunLon - dphi - obsLon)%twopi ### we pick the Western intersection as the sunrise (so that the sun is to the East of the observatory at this time)

def dAngUntilSunset( obsLat, obsLon, sunLat, sunLon, obsAng, solarOcclusionAng=48*deg2rad ):
    """
    returns hour angle in radians
    """
    obsTheta = pi2 - obsLat
    sunTheta = pi2 - sunLat
    A = np.cos(obsTheta)*np.cos(sunTheta)
    B = np.sin(obsTheta)*np.sin(sunTheta)

    dphi = np.arccos( (np.cos(obsAng+solarOcclusionAng) - A)/B ) ### angular distance relative to the sun when the intersection will happen
    return (sunLon + dphi - obsLon)%twopi ### we pick the Eastern intersection as the sunset (so that the sun is to the West of the observatory at this time)

def dAngUntilRises( obsLat, obsLon, srcDec, srcRA, gps, obsAng ):
    """
    returns hour angle in radians
    """
    obsTheta = pi2 - obsLat
    srcTheta = pi2 - srcDec
    srcPhi = rotateRAC2E( srcRA, gps ) 

    A = np.cos(obsTheta)*np.cos(srcTheta)
    B = np.sin(obsTheta)*np.sin(srcTheta)

    cosObsAng = np.cos(obsAng)
    if cosObsAng < A + B*np.cos(obsLon-srcPhi): ### currently observable
        return 0 

    dphi = np.arccos( (cosObsAng - A)/B ) ### angular distance relative to the src when the intersection will happen
    return (srcPhi - dphi - obsLon)%twopi ### we pick the Western intersection so that the source is to the East of the observatory at this time and therefore has just risen

def array_dAngUntilRises( obsLat, obsLon, srcDec, srcRA, gps, obsAng ):
    """
    returns hour angle in radians

    expects srcDec and srcRA to be array objects
    """
    obsTheta = pi2 - obsLat
    srcTheta = pi2 - srcDec
    srcPhi = rotateRAC2E( srcRA, gps )

    A = np.cos(obsTheta)*np.cos(srcTheta)
    B = np.sin(obsTheta)*np.sin(srcTheta)

    cosObsAng = np.cos(obsAng)
    ans = np.empty_like(srcDec)

    truth = cosObsAng <= A + B*np.cos(obsLon-srcPhi) ### currently observable
    ans[truth] = 0

    truth = (1-truth).astype(bool) ### select off only those positions for which dphi is defined and fill them in
    dphi = np.arccos( (cosObsAng - A[truth])/B[truth] ) ### angular distance relative to the src when the intersection will happen

    ans[truth] = (srcPhi - dphi - obsLon)%twopi
    return ans

def timeUntilObservable( obsLat, obsLon, srcDec, srcRA, sunDec, sunRA, gps, obsAng, solarOcclusionAng=48*deg2rad):
    """
    computes the time until the source is observable from (obsLat,obsLon)

    does NOT assume the source is observable and checks explicitly.

    all angles should be specified in radians

    could perhaps be optimized a bit more, but that mostly invovles passing values via delegation so the helper functions don't have to re-compute values
    """
    # get solar position in spherical coordinate
    sunTheta = pi2 - sunDec
    cosSunTheta = np.cos(sunTheta)
    sinSunTheta = np.sin(sunTheta)
    sunPhi = rotateRAC2E( sunRA, gps )

    ### compute a few useful quantities
    obsTheta = pi2-obsLat
    cosObsTheta = np.cos(obsTheta)
    sinObsTheta = np.sin(obsTheta)

    srcTheta = pi2 - srcDec
    cosTheta = np.cos(srcTheta)
    sinTheta = np.sin(srcTheta)
    srcPhi   = rotateRAC2E( srcRA, gps )

    cosObsAng = np.cos(obsAng)
    cosObsAngSOA = np.cos(solarOcclusionAng+obsAng)

    #---------------------------------------------
    ### check whether the source will ever be observable
    #---------------------------------------------
    ### compute the angular distance between phi_sun and the effective terminator
    cosdphi = (cosObsAngSOA - cosSunTheta*cosObsTheta)/(sinSunTheta*sinObsTheta)

    if cosdphi < -1: ### polar day -> never can observer
        return np.infty

    elif cosdphi > 1: ### polar night -> always can observe
        if np.abs(obsLat-srcDec) <= obsAng:  ### check whether the change in latitude is small enough
            return 0 ### we can observe immediately
        else:
            return np.infty ### can never observe it

    else: ### there is a sunrise and a sunset  
        ### there will be times when this latitude can observe, so we figure out the change in RA associated therewith 
        dphi = np.arccos( cosdphi )

        ### compute a few helpful numbers
        A = cosTheta*cosObsTheta
        B = sinTheta*sinObsTheta

        ### check to see whether source is observable at sunrise or sunset
        obsAtSunset  = cosObsAng <= A + B*np.cos( sunPhi+dphi - srcPhi ) 
        obsAtSunrise = cosObsAng <= A + B*np.cos( sunPhi-dphi - srcPhi )

        ### check that it is within the latitude bounds
        withinLat = (obsTheta-obsAng<=srcTheta)*(srcTheta<=obsTheta+obsAng)
        ### check that it is within the longitude bounds
        withinLon = abs(sunPhi%twopi - srcPhi%twopi) >= dphi

        if not (obsAtSunset or obsAtSunrise or (withinLat and withinLon)):
            return np.infty # never observable

    #---------------------------------------------
    ### if we have not exited, yet, then we know it must be observable at some time.
    #---------------------------------------------
    ### is the observatory currently in daylight?
    if cosObsAngSOA < cosObsTheta*cosSunTheta + sinObsTheta*sinSunTheta*np.cos(obsLon-sunPhi): ### currently in daylight
        ### find hour-angle until sunset
        dobsLon = dAngUntilSunset( obsLat, obsLon, sunDec, sunPhi, obsAng, solarOcclusionAng=solarOcclusionAng )
        ### check if source is observable at sunset
        if cosObsAng < cosObsTheta*cosTheta + sinObsTheta*sinTheta*np.cos(obsLon+dobsLon - srcPhi): ### observable at sunset
            return dobsLon*43200/np.pi ### return the time until sunset
        else: ### not observable at sunset, so we have to wait for it to rise
            dAng = dAngUntilRises( obsLat, obsLon+dobsLon, srcDec, srcRA, gps, obsAng ) ### angle (after sunset) before source rises
            return (dAng + dobsLon)*43200/np.pi ### return time until sunset + time after sunset before source rises

    else: ### NOT currenlty in daylight
        ### can we observe it now?
        if cosObsAng < cosObsTheta*cosTheta + sinObsTheta*sinTheta*np.cos(obsLon-srcPhi): ### we can observe it now
            return 0
        else: ### we can NOT observe it now
            ### find hour-angle until sunrise
            dobsLon = dAngUntilSunrise( obsLat, obsLon, sunDec, sunPhi, obsAng, solarOcclusionAng=solarOcclusionAng )
            ###              observable some time this night                          we can observe it at sunrise, catch the bit the last conditional missed
            if ((obsLon < srcPhi) and (srcPhi <= obsLon+dobsLon)) or (cosObsAng < cosObsTheta*cosTheta + sinObsTheta*sinTheta*np.cos(obsLon+dobsLon - srcPhi)):
                return dAngUntilRises( obsLat, obsLon, srcDec, srcRA, gps, obsAng )*43200/np.pi ### return time until src rises

            #-------------------------------------
            ### if we get here, we know the sun must rise and set before we can see the source
            #-------------------------------------
            ### find hour-angle until sunset
            dobsLon = dAngUntilSunset( obsLat, obsLon, sunDec, sunPhi, obsAng, solarOcclusionAng=solarOcclusionAng )
            if cosObsAng < cosObsTheta*cosTheta + sinObsTheta*sinTheta*np.cos(obsLon+dobsLon - srcPhi): ### we can observe it at sunset
                return dobsLon*43200/np.pi ### return time until sunset
            else:
                return dAngUntilRises( obsLat, obsLon, srcDec, srcRA, gps, obsAng )*43200/np.pi ### return time until src rises (should be more than time until sunset)

def marge_timeUntilObservable( post, obsLat, obsLon, sunDec, sunRA, gps, obsAng, solarOcclusionAng=48*deg2rad ):
    """
    computes average of timeUntilObservable with weighting by the skymap

    return Real -> using only those parts of the sky that are observable
           Imag -> the amount of probability that is not observable

    NOTE: we do not pass a 'mask' because this function computes that itself
    """

    ### get solar position in spherical coordinate
    sunTheta = pi2 - sunDec
    cosSunTheta = np.cos(sunTheta)
    sinSunTheta = np.sin(sunTheta)
    sunPhi = rotateRAC2E( sunRA, gps )

    ### compute a few useful quantities about the observatory
    obsTheta = pi2-obsLat
    cosObsTheta = np.cos(obsTheta)
    sinObsTheta = np.sin(obsTheta)

    ### compute useful quantities about the skymap
    
    npix = len(post)
    nside = hp.npix2nside( npix )
    pix = np.arange(npix, dtype=int)
    srcTheta, srcRA = hp.pix2ang( nside, pix )
    srcDec = pi2 - srcTheta
    srcPhi = rotateRAC2E( srcRA, gps )
    cosTheta = np.cos(srcTheta)
    sinTheta = np.sin(srcTheta)

    cosObsAng = np.cos(obsAng)
    cosObsAngSOA = np.cos(solarOcclusionAng+obsAng)

    ### the averaged value we compute
    ### Real part is the average value, computed using only the parts of the sky that are actually observable
    ### Imag part is the probability that is NOT observable (without the delay time, because that would make it infinite)
    delayTime = np.empty((npix,), dtype=complex)

    #---------------------------------------------
    ### check whether the source will ever be observable
    #---------------------------------------------
    ### compute the angular distance between phi_sun and the effective terminator
    cosdphi = (cosObsAngSOA - cosSunTheta*cosObsTheta)/(sinSunTheta*sinObsTheta)

    if cosdphi < -1: ### polar day -> never can observer
        return 0.0 + 1.0j

    elif cosdphi > 1: ### polar night -> always can observe
        mask = np.abs(obsLat-srcDec) <= obsAng ### check whether the change in latitude is small enough
        ###   the part of the sky that is observable is immediately observable -> Real = 0
        ###   the probability that is not observable is given by np.sum(post*(1-mask))
        return 0.0 + np.sum( post*(1-mask) )*1.0j

    else: ### there is a sunrise and a sunset  
        ### there will be times when this latitude can observe, so we figure out the change in RA associated therewith 
        dphi = np.arccos( cosdphi )

        ### compute a few helpful numbers
        A = cosTheta*cosObsTheta
        B = sinTheta*sinObsTheta

        ### check to see whether source is observable at sunrise or sunset
        obsAtSunset  = cosObsAng <= A + B*np.cos( sunPhi+dphi - srcPhi )
        obsAtSunrise = cosObsAng <= A + B*np.cos( sunPhi-dphi - srcPhi )

        ### check that it is within the latitude bounds
        withinLat = (obsTheta-obsAng<=srcTheta)*(srcTheta<=obsTheta+obsAng)
        ### check that it is within the longitude bounds
        withinLon = np.abs(sunPhi%twopi - srcPhi%twopi) >= dphi

        ### the total observable region
        observable = (obsAtSunset + obsAtSunrise + withinLat*withinLon) > 0 ### true when we can observe at some point
        delayTime[(1-observable).astype(bool)] = 1.0j ### this part of the sky is not observable so it gets an imaginary value
                                       ### the rest of this function should only add Real numbers to delayTime...

    #---------------------------------------------
    ### beyond this point, we know the remaining source locations must be observable at some time.
    ### those indicies are stored in 'observable'
    #---------------------------------------------

    ### is the observatory currently in daylight?
    if cosObsAngSOA < cosObsTheta*cosSunTheta + sinObsTheta*sinSunTheta*np.cos(obsLon-sunPhi): ### currently in daylight
        ### find hour-angle until sunset
        dobsLon = dAngUntilSunset( obsLat, obsLon, sunDec, sunPhi, obsAng, solarOcclusionAng=solarOcclusionAng )

        ### check if source is observable at sunset
        obsAtSunset = cosObsAng < cosObsTheta*cosTheta + sinObsTheta*sinTheta*np.cos(obsLon+dobsLon - srcPhi)

        delayTime[obsAtSunset*observable] = dobsLon*43200/np.pi ### select off just the prob that is observable at Sunset and assign the time until sunset

        ### add the delay time for prob that is not observable at Sunset (we have to wait for it to rise)
        waitUntilRises = observable*(1-obsAtSunset).astype(bool) ### things that are observable but which are not observable at Sunset
        dAng = array_dAngUntilRises( obsLat, obsLon+dobsLon, srcDec[waitUntilRises], srcRA[waitUntilRises], gps, obsAng ) ### angle (after sunset) before source rises

        delayTime[waitUntilRises] = (dAng+dobsLon)*43200/np.pi ### select off just the probability that is observable but not at Sunset and add it up

    else: ### NOT currenlty in daylight

        ### can we observe it now?
        obsNow = cosObsAng <= cosObsTheta*cosTheta + sinObsTheta*sinTheta*np.cos(obsLon-srcPhi)

        delayTime[observable*obsNow] = 0.0

        ### add contribution for sources we can NOT observe now

        ### find hour-angle until sunrise
        dobsLon = dAngUntilSunrise( obsLat, obsLon, sunDec, sunPhi, obsAng, solarOcclusionAng=solarOcclusionAng )

        ###              observable some time this night                          we can observe it at sunrise, catch the bit the last conditional missed
        obsBeforeSunrise = ((obsLon < srcPhi)*(srcPhi <= obsLon+dobsLon)) + (cosObsAng < cosObsTheta*cosTheta + sinObsTheta*sinTheta*np.cos(obsLon+dobsLon - srcPhi)) > 0 ### we can observe it at some point before sunrise
        
        truth = observable*(1-obsNow).astype(bool)*obsBeforeSunrise ### things that are observable, but not observable now and will be observable some time before Sunrise
        dAng = array_dAngUntilRises( obsLat, obsLon, srcDec[truth], srcRA[truth], gps, obsAng)

        delayTime[truth] = dAng*43200/np.pi ### return time until src rises

        ### the rest: the sun must rise and set before we can see the source
        theRest = observable*((1-obsBeforeSunrise)*(1-obsNow)).astype(bool) ### observable, but not observable now and not observable before Sunrise

        ### find hour-angle until sunset
        dobsLon = dAngUntilSunset( obsLat, obsLon, sunDec, sunPhi, obsAng, solarOcclusionAng=solarOcclusionAng )

        ### add the stuff that is observable at Sunset
        obsAtSunset = cosObsAng < cosObsTheta*cosTheta + sinObsTheta*sinTheta*np.cos(obsLon+dobsLon - srcPhi) ### we can observe it at Sunset

        delayTime[theRest*obsAtSunset] = dobsLon*43200/np.pi ### positions that have not been assigned a value yet and are observable at Sunset

        ### add the rest
        truth = theRest*(1-obsAtSunset).astype(bool)
        dAng = array_dAngUntilRises( obsLat, obsLon, srcDec[truth], srcRA[truth], gps, obsAng )

        delayTime[truth] = dAng*43200/np.pi 

    return np.sum( post*delayTime ) ### take the integral -> weighted average

#-------------------------------------------------
### depricated because of slow function call to find solar and lunar positions
#-------------------------------------------------

def timeUntilUnoccluded( srcDec, srcRA, gps, solarOcclusionAng, lunarOcclusionAng, absErr=1 ):
    """
    computes the amount of time before a source (located at srcDec, srcRA) arriving at gps is not occluded by either the Sun or the Moon, assuming the occlusion angles specified
    performs a bisection search until the time is known to better than "absErr" seconds
    """
    ### check to see if it is currently occluded
    if not (isSolarOccluded( srcDec, srcRA, gps, solarOcclusionAng ) + isLunarOccluded( srcDec, srcRA, gps, lunarOcclusionAng )):
        return 0.

    ### generate one per month and evaluate all of these
    gps_sample = np.linspace(gps, gps+31449600, 25) ### sample two per month
    occ = isSolarOccluded( srcDec, srcRA, gps_sample, solarOcclusionAng ) + isLunarOccluded( srcDec, srcRA, gps_sample, lunarOcclusionAng )

    ### find out the first month it is no occluded
    good = np.nonzero(1-occ)[0][0] ### earliest time visible

    ### extract the corresponding gps times
    bad = gps_sample[good-1]
    good = gps_sample[good]

    ### bisection search until error<absErr
    while np.abs(good-bad) > absErr:
        guess = 0.5*(good+bad)
        if isSolarOccluded( srcDec, srcRA, guess, solarOcclusionAng ) + isLunarOccluded( srcDec, srcRA, guess, lunarOcclusionAng ):
            bad = guess
        else:
            good = guess

    return 0.5*(good+bad) - gps

def timeUntilSolarUnoccluded( srcDec, srcRA, gps, solarOcclusionAng, absErr=1 ):
    """
    computes the amount of time before a source (located at srcDec, srcRA) arriving at gps is not occluded by the Sun, assuming the occlusion angles specified
    performs a bisection search through time
    absErr is the maximum allowable error in absolute time (seconds) for the answer
    """
    ### check to see if it is currently occluded
    if not isSolarOccluded( srcDec, srcRA, gps, solarOcclusionAng ):
        return 0.

    ### generate one per month and evaluate all of these
    gps_sample = np.linspace(gps, gps+364*86400, 25) ### sample two per month
    occ = isSolarOccluded( srcDec, srcRA, gps_sample, solarOcclusionAng )

    ### find out the first month it is no occluded
    good = np.nonzero(1-occ)[0][0] ### earliest time here

    ### extract the corresponding gps times
    bad = gps_sample[good-1]
    good = gps_sample[good]

    ### bisection search until error<absErr
    while np.abs(good-bad) > absErr:
        guess = 0.5*(good+bad)
        if isSolarOccluded( srcDec, srcRA, guess, solarOcclusionAng ):
            bad = guess
        else:
            good = guess

    return 0.5*(good-bad) - gps
        
def isSolarOccluded( dec, ra, gps, dead_zone ):
    if isinstance( gps, (np.ndarray, list, tuple) ):
        theta_sun = np.empty_like( gps )
        phi_sun = np.empty_like( gps )
        for i, t in enumerate(gps):
            timeObj = astropyTime( tconvert(int(t), form="%Y-%m-%dT%H:%M:%S")+("%.3f"%(t%1))[1:], format='isot', scale='utc')

            ### get solar position in spherical coordinate
            sun = astropyCoordinates.get_sun(timeObj)
            theta_sun[i] = pi2 - sun.dec.radian
            phi_sun[i] = sun.ra.radian
    else:
        timeObj = astropyTime( tconvert(int(gps), form="%Y-%m-%dT%H:%M:%S")+("%.3f"%(gps%1))[1:], format='isot', scale='utc')

        ### get solar position in spherical coordinate
        sun = astropyCoordinates.get_sun(timeObj)
        theta_sun = pi2 - sun.dec.radian
        phi_sun = sun.ra.radian

    theta = pi2 - dec
    ### compute cos(theta) between all pixels and the sun in spherical coordinates
    cosdtheta = np.cos(theta_sun)*np.cos(theta) + np.sin(theta_sun)*np.sin(theta)*np.cos(phi_sun-ra)

    return cosdtheta > np.cos(dead_zone)

    
def timeUntilLunarUnoccluded( srcDec, srcRA, gps, lunarOcclusionAng, absErr=1 ):
    """
    computes the amount of time before a source (located at srcDec, srcRA) arriving at gps is not occluded by the Moon, assuming the occlusion angles specified
    performs a bisection search through time
    """
    if not isLunarOccluded( srcDec, srcRA, gps, lunarOcclusionAng ):
        return 0.

    ### generate one per month and evaluate all of these
    gps_sample = np.linspace(gps, gps+364*86400, 25) ### sample two per month
    occ = isLunarOccluded( srcDec, srcRA, gps_sample, lunarOcclusionAng )

    ### find out the first month it is no occluded
    good = np.nonzero(1-occ)[0][0] ### earliest time here

    ### extract the corresponding gps times
    bad = gps_sample[good-1]
    good = gps_sample[good]

    ### bisection search until error<absErr
    while np.abs(good-bad) > absErr:
        guess = 0.5*(good+bad)
        if isLunarOccluded( srcDec, srcRA, guess, lunarOcclusionAng ):
            bad = guess
        else:
            good = guess

    return 0.5*(good-bad) - gps

def isLunarOccluded( dec, ra, gps, dead_zone ):
    moon = ephem.Moon()
    if isinstance( gps, (np.ndarray, list, tuple) ):
        theta_moon = np.empty_like(gps)
        phi_moon = np.empty_like(gps)
        for i, t in enumerate(gps):
            moon.compute(tconvert(int(t), form="%Y/%m/%d %H:%M:%S"))
            theta_moon[i] = pi2 - moon.dec
            phi_moon[i] = moon.ra
    else:
        moon.compute(tconvert(int(gps), form="%Y/%m/%d %H:%M:%S"))
        theta_moon = pi2 - moon.dec
        phi_moon = moon.ra

    theta = pi2 - dec
    ### compute cos(theta) between all pixels and the sun in spherical coordinates
    cosdtheta = np.cos(theta_moon)*np.cos(theta) + np.sin(theta_moon)*np.sin(theta)*np.cos(phi_moon-ra)

    return cosdtheta > np.cos(dead_zone)
