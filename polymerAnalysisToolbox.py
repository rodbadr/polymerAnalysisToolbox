import numpy as np
import gsd
from packaging import version
import gsd.hoomd
import sys

pi = np.pi

""" TODO: 
            -Add density profile fitting
            -Write test script to test the functions in this module
"""

def computeCOM_periodic(x_i, L_i):

    """ 
    Compute the center of mass for a periodic system
    using the angle method. The coordinates need to be in the range [0, L_i).
    x_i: positions of the particles in one dimension
    L_i: box length in the corresponding direction
    """

    th = x_i / L_i * 2 * pi

    Xi = np.cos(th)

    Zeta = np.sin(th)

    Xi_bar = Xi.mean()

    Zeta_bar = Zeta.mean()

    th_bar = np.arctan2(-Zeta_bar, -Xi_bar) + pi

    com = L_i * th_bar / 2 / pi

    return com

def computeCOM(x_i):

    """
    Compute the center of mass the usual way
    x_i: positions of the particles in one dimensions
    """
    com = np.mean(x_i, axis=0)

    return com

def centerCOM_periodic(positions, Lx, Ly, Lz, originAtCenter=True):
    """
    Place the center of mass of the system at the center in a periodic box.
    positions: positions of the particles
    Lx, Ly, Lz: box dimensions
    originAtCenter: if True, the origin of the coordinate system is at the center of the box.
    """
    if originAtCenter:
        com = np.array([computeCOM_periodic(positions[:, 0] + Lx/2.0, Lx) - Lx/2.0,
                        computeCOM_periodic(positions[:, 1] + Ly/2.0, Ly) - Ly/2.0,
                        computeCOM_periodic(positions[:, 2] + Lz/2.0, Lz) - Lz/2.0])

        centered_positions = positions.copy()
        centered_positions[:, 0]  = (centered_positions[:, 0] + Lx/2.0 - com[0]) % Lx - Lx/2.0
        centered_positions[:, 1]  = (centered_positions[:, 1] + Ly/2.0 - com[1]) % Ly - Ly/2.0
        centered_positions[:, 2]  = (centered_positions[:, 2] + Lz/2.0 - com[2]) % Lz - Lz/2.0

    else:

        com = np.array([computeCOM_periodic(positions[:, 0], Lx),
                        computeCOM_periodic(positions[:, 1], Ly),
                        computeCOM_periodic(positions[:, 2], Lz)])

        centered_positions = positions - com + np.array([Lx/2.0, Ly/2.0, Lz/2.0])
        centered_positions[:, 0]  = centered_positions[:, 0] % Lx
        centered_positions[:, 1]  = centered_positions[:, 1] % Ly
        centered_positions[:, 2]  = centered_positions[:, 2] % Lz

    return centered_positions


def computeRg_periodic_old(positions, Nchains, chainLength, Lx, Ly, Lz):
    """
    Compute the radius of gyration for a set of chains.
    positions: positions of the particles
    Nchains: number of chains
    chainLength: length of each chain
    Lx, Ly, Lz: box dimensions
    """
    Rg = np.zeros(Nchains)

    new_positions = wholeChains(positions, Nchains, chainLength, Lx, Ly, Lz)

    for chain in range(Nchains):
        start = chain * chainLength
        end = start + chainLength
        com = np.array([computeCOM(new_positions[start:end,0]),
                        computeCOM(new_positions[start:end,1]),
                        computeCOM(new_positions[start:end,2])])
        Rg[chain] = np.sqrt(np.mean(np.sum((new_positions[start:end] - com) ** 2, axis=1)))

    return Rg


def computeRg_periodic(positions, Nchains, chainLength, Lx, Ly, Lz, originAtCenter=True):
    """
    Compute the radius of gyration for a set of chains in periodic boundary conditions where the origin is at the center.
    positions: positions of the particles
    Nchains: number of chains
    chainLength: length of each chain
    Lx, Ly, Lz: box dimensions
    originAtCenter: if True, the origin of the coordinate system is centered at the center of the box.
    """
    Rg = np.zeros(Nchains)

    for chain in range(Nchains):
        start = chain * chainLength
        end = start + chainLength

        if originAtCenter:
            com = np.array([computeCOM_periodic(positions[start:end,0] + Lx/2.0, Lx) - Lx/2.0,
                            computeCOM_periodic(positions[start:end,1] + Ly/2.0, Ly) - Ly/2.0,
                            computeCOM_periodic(positions[start:end,2] + Lz/2.0, Lz) - Lz/2.0])
        else:
            com = np.array([computeCOM(positions[start:end,0]),
                            computeCOM(positions[start:end,1]),
                            computeCOM(positions[start:end,2])])
        
        diff = positions[start:end].copy()

        diff[:, 0] = (diff[:, 0] - com[0]) % Lx
        diff[:, 1] = (diff[:, 1] - com[1]) % Ly
        diff[:, 2] = (diff[:, 2] - com[2]) % Lz

        mask = (diff[:,0] > Lx / 2.0)
        diff[mask, 0] -= Lx
        mask = (diff[:,1] > Ly / 2.0)
        diff[mask, 1] -= Ly
        mask = (diff[:,2] > Lz / 2.0)
        diff[mask, 2] -= Lz
    
        Rg[chain] = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    return Rg


def computeDensityProfile_3D(positions, DX, DY, DZ, Lx, Ly, Lz, originAtCenter=True):
    """
    Compute the density profile for a set of particles.
    positions: positions of the particles
    DX, DY, DZ: bin sizes in each dimension. Set to 0 to use the entire box length
    Lx, Ly, Lz: box dimensions
    originAtCenter: if True, the origin is at the center of the box
    """

    if DX == 0:
        DX = Lx
    if DY == 0:
        DY = Ly
    if DZ == 0:
        DZ = Lz

    if originAtCenter:
        if min(positions[:, 0]) < -Lx/2 or max(positions[:, 0]) > Lx/2 or \
           min(positions[:, 1]) < -Ly/2 or max(positions[:, 1]) > Ly/2 or \
           min(positions[:, 2]) < -Lz/2 or max(positions[:, 2]) > Lz/2:
            
            print("Warning: positions are outside the box dimensions.")

        hist, edges = np.histogramdd(positions, bins=[np.arange(-Lx/2, Lx/2+DX*1e-8, DX), np.arange(-Ly/2, Ly/2+DY*1e-8, DY), np.arange(-Lz/2, Lz/2+DZ*1e-8, DZ)], density=False)

    else:

        if min(positions[:, 0]) < 0 or max(positions[:, 0]) > Lx or \
           min(positions[:, 1]) < 0 or max(positions[:, 1]) > Ly or \
           min(positions[:, 2]) < 0 or max(positions[:, 2]) > Lz:
            
            print("Warning: positions are outside the box dimensions.")

        hist, edges = np.histogramdd(positions, bins=[np.arange(0, Lx+DX*1e-8, DX), np.arange(0, Ly+DY*1e-8, DY), np.arange(0, Lz+DZ*1e-8, DZ)],density=False)

    # Normalize histogram to get density
    density = hist / (DX * DY * DZ)

    return density


def computeDensityProfile_1D(positions, Di, Lx, Ly, Lz, dim = 2, originAtCenter=True):
    """
    Compute the 1D density profile for a set of particles.
    positions: positions of the particles
    Di: bin sizes in the desired dimension.
    Lx, Ly, Lz: box dimensions
    dim: dimension to compute the density profile
    originAtCenter: if True, the origin is at the center of the box
    """

    if dim == 0:
        Li = Lx
        Area = Ly * Lz
    elif dim == 1:
        Li = Ly
        Area = Lx * Lz
    elif dim == 2:
        Li = Lz
        Area = Lx * Ly

    if originAtCenter:
        if min(positions[:, dim]) < -Li/2 or max(positions[:, dim]) > Li/2:

            print("Warning: positions are outside the box dimensions.")

        hist, edges = np.histogram(positions[:, dim], bins=np.arange(-Li/2, Li/2+Di*1e-8, Di), density=False)

    else:

        if min(positions[:, dim]) < 0 or max(positions[:, dim]) > Li:

            print("Warning: positions are outside the box dimensions.")

        hist, edges = np.histogramd(positions, bins=np.arange(0, Li+Di*1e-8, Di),density=False)

    # Normalize histogram to get density
    density = hist / (Di * Area)

    return density


def wholeChains(positions, Nchains, chainLength, Lx, Ly, Lz, unwrapX=True, unwrapY=True, unwrapZ=True):
    """
    Make chains whole by unwrapping bonds that cross the periodic boundary.
    positions: positions of the particles
    Nchains: number of chains
    chainLength: length of each chain
    Lx, Ly, Lz: box dimensions
    unwrapX, unwrapY, unwrapZ: if True, unwrap the positions in the corresponding direction
    """
    # Make a copy to avoid modifying original input
    new_positions = positions.copy()

    for chain in range(Nchains):
        for mon in range(chainLength - 1):
            i = chain * chainLength + mon
            j = i + 1

            dx = new_positions[i, 0] - new_positions[j, 0]
            dy = new_positions[i, 1] - new_positions[j, 1]
            dz = new_positions[i, 2] - new_positions[j, 2]

            if (abs(dx) > Lx / 2) and unwrapX:
                temp = new_positions[chain*chainLength:(chain+1)*chainLength, 0].copy()
                temp[mon+1:] += np.sign(dx) * Lx
                new_positions[chain*chainLength:(chain+1)*chainLength, 0] = temp

            if (abs(dy) > Ly / 2) and unwrapY:
                temp = new_positions[chain*chainLength:(chain+1)*chainLength, 1].copy()
                temp[mon+1:] += np.sign(dy) * Ly
                new_positions[chain*chainLength:(chain+1)*chainLength, 1] = temp

            if (abs(dz) > Lz / 2) and unwrapZ:
                temp = new_positions[chain*chainLength:(chain+1)*chainLength, 2].copy()
                temp[mon+1:] += np.sign(dz) * Lz
                new_positions[chain*chainLength:(chain+1)*chainLength, 2] = temp

    return new_positions


def unwrap_while_whole(positions, images, ref_pos, ref_images, ref_frame_whole, Lx, Ly, Lz):
    """
    Shift positions by the reference position accounting for image data
    and make sure the chains are whole.
    positions: positions of the particles
    images: images of the particles
    ref_pos: reference positions of the particles (first frame we are considering)
    ref_images: reference images of the particles (first frame we are considering)
    ref_frame_whole: positions of the particles in the first frame, made whole
    Lx, Ly, Lz: box dimensions
    """
    dx = positions[:, 0] + images[:, 0] * Lx - (ref_pos[:, 0] + ref_images[:, 0] * Lx)
    dy = positions[:, 1] + images[:, 1] * Ly - (ref_pos[:, 1] + ref_images[:, 1] * Ly)
    dz = positions[:, 2] + images[:, 2] * Lz - (ref_pos[:, 2] + ref_images[:, 2] * Lz)

    new_positions = np.empty_like(positions)
    new_positions[:, 0] = ref_frame_whole[:, 0] + dx
    new_positions[:, 1] = ref_frame_whole[:, 1] + dy
    new_positions[:, 2] = ref_frame_whole[:, 2] + dz

    return new_positions

def validate_positions(positions, dir, fileroot, frame):
    """
    Validate positions by comparing with a reference trajectory.
    positions: positions of the particles
    dir: directory of the reference trajectory
    fileroot: root name of the reference trajectory file
    frame: frame number to validate
    """
    fname_validation = dir + fileroot

    t_validation = openGSDTrajectory(fname_validation)

    ref_pos = t_validation[frame].particles.position[:].copy()

    validation = positions - ref_pos
    validation = validation.sum()
    
    if abs(validation) > 1e-6:
        print(f"Validation failed for frame {frame}. Difference: {validation}")
        sys.exit(1)

def openGSDTrajectory(fname):
    """
    Open a GSD trajectory file and return the trajectory object.
    accounts for different GSD versions.
    """

    # Try to get 'version' attribute, fallback to '__version__'
    gsd_version = getattr(gsd.version, 'version', getattr(gsd.version, '__version__', None))

    if gsd_version is None:
        raise RuntimeError("Could not determine GSD version.")

    gsd_version = version.parse(gsd_version)
    open_mode = 'r' if gsd_version > version.parse("2.9.0") else 'rb'
    t = gsd.hoomd.open(name=fname, mode=open_mode)

    return t


def nonUniformFT_brute(times, signal, n_freqs=100, maxFreq=None, freqs=None, integrate=False):
    """
    Compute the non-uniform Fourier transform (NUFT) of a signal using a brute-force approach.

    Parameters:
    - times: 1D array of time points (must be sorted)
    - signal: 1D array of signal values at those time points
    - n_freqs: number of frequency points to compute (default: 100)
    - maxFreq: maximum frequency to consider (optional, defaults to Nyquist frequency computed below)
    - freqs: 1D array of frequency points (optional)
    - integrate: if True, multiply the difference in time points before summing (default: False)

    Returns:
    - freqs: 1D array of frequency points
    - transform: 1D array of complex Fourier coefficients

    NOTE 1: This calculates only in the positive frequency domain. 
            This is only sufficient for real valued signals, not complex ones

    NOTE 2: if the time points are in fact uniform, the frequency points are ideally sampled
            from an array of the form: np.linspace(dt_min0, fs, len(times)/2)
            where fs is the Nyquist (sampling) frequency: fs = 1/(2*dt_min) 
            with dt_min the time interval between samples. At frequencies
            not belonging to this array, spurious artifacts may appear.
    """
    if freqs is None: # compute with a uniform grid of frequencies
        if n_freqs > len(times)//2:
            print("Warning: n_freqs is larger than half the number of time points. This may lead to aliasing.")

        if maxFreq is None: # if no maximum frequency is provided, use the Nyquist frequency as the maximum
            maxFreq = 1/(2*np.min(np.diff(times)))

        freqs = np.linspace(0, maxFreq, n_freqs)

        # Compute NUFT
        transform = np.zeros(n_freqs, dtype=complex)
        for k, f in enumerate(freqs):
            if integrate:
                dt_temp = np.diff(times, prepend=0)
                transform[k] = np.sum(signal * np.exp(-2j * np.pi * f * times) * dt_temp)
            else:
                transform[k] = np.sum(signal * np.exp(-2j * np.pi * f * times))

        return freqs, transform
    
    else: # if an array of frequencies is provided, use the frequencies in the array

        # Compute NUFT
        transform = np.zeros(len(freqs), dtype=complex)
        for k, f in enumerate(freqs):
            if integrate:
                dt_temp = np.diff(times, prepend=0)
                transform[k] = np.sum(signal * np.exp(-2j * np.pi * f * times) * dt_temp)
            else:
                transform[k] = np.sum(signal * np.exp(-2j * np.pi * f * times))

        return freqs, transform


def nonUniformFT_fast(times, signal, n_freqs=100, freqs=None, integrate=False):
    """
    Compute and plot the NUFFT spectrum of a non-uniformly sampled signal.

    Parameters:
    - times: 1D array of non-uniform sampling times
    - signal: 1D array of signal values at those times
    - n_freqs: number of frequency bins (default 100)
    - freqs: 1D array of frequency points (optional)
    - integrate: if True, multiply the difference in time points before transforming (default: False)

    NOTE: if the time points are in fact uniform, the frequency points are ideally sampled
          from an array of the form: np.linspace(dt_min0, fs, len(times)/2)
          where fs is the Nyquist (sampling) frequency: fs = 1/(2*dt_min) 
          with dt_min the time interval between samples. At frequencies
          not belonging to this array, spurious artifacts may appear.
    """

    try:
        import finufft

    except ImportError:
        print("ERROR: finufft is not installed. Please install it to use this function.\n To install you can use: pip install finufft")
        sys.exit(1)

    # Rescale time to [-π, π] as required by finufft
    t_min = np.min(times)
    t_max = np.max(times)
    t_scaled = 2 * np.pi * (times - t_min) / (t_max - t_min) - np.pi

    if freqs is None: # Compute fiNUFFT type 1: non-uniform time to uniform frequency

        if integrate:
            dt_temp = np.diff(times, prepend=0)
            signal = signal * dt_temp
        
        transform = finufft.nufft1d1(t_scaled, signal.astype(np.complex128), n_freqs)

        # finufft type 1 works by dividing the time domain into equal bins of size dt
        # then assigning the frequencies
        dt = (t_max - t_min)/n_freqs

        # Compute corresponding frequencies
        freqs = np.fft.fftfreq(n_freqs, d=dt)
        freqs = np.fft.fftshift(freqs)  # Center zero frequency

        return freqs, transform

    else: # Compute fiNUFFT type 3: non-uniform time to specific frequencies

        if integrate:
            dt_temp = np.diff(times, prepend=0)
            signal = signal * dt_temp

        dt_min = np.min(np.diff(np.sort(times)))
        tempNmeas = t_max/dt_min
        tempFreqs = freqs*tempNmeas*dt_min # scale frequencies to the range used by finufft (index based)

        transform = finufft.nufft1d3(t_scaled, signal.astype(np.complex128), tempFreqs)

        return freqs, transform

