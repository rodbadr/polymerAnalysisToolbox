import numpy as np
import gsd
from packaging import version
import gsd.hoomd
import sys

pi = np.pi

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
    Center the center of mass of the system in a periodic box.
    positions: positions of the particles
    Lx, Ly, Lz: box dimensions
    originAtCenter: if True, the origin of the coordinate system is centered at the center of the box.
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

        centered_positions = positions - com
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
    DX, DY, DZ: bin sizes in each dimension
    Lx, Ly, Lz: box dimensions
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

        # Create 2D histogram
        hist, edges = np.histogramdd(positions, bins=[np.arange(0, Lx+DX*1e-8, DX), np.arange(0, Ly+DY*1e-8, DY), np.arange(0, Lz+DZ*1e-8, DZ)],density=False)

    # Normalize histogram to get density
    density = hist / (DX * DY * DZ)

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
    Shift positions by the reference position accounting for image
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

def writeNewTrajectory_unwrapped_and_whole(file_in, file_out, ref_frame_whole, Lx, Ly, Lz):
    """
    Write a new GSD trajectory with unwrapped and whole chains.
    t: trajectory object
    fname: filename to write the new trajectory
    ref_frame_whole: reference frame with whole chains
    Lx, Ly, Lz: box dimensions
    """
    # Load original trajectory
    with gsd.hoomd.open(file_in, 'rb') as traj:
        total_frames = len(traj)
        print(f"Original number of frames: {total_frames}")

        # Floor to nearest multiple of 100
        rounded_frames = int(np.floor(total_frames / 100)) * 100
        print(f"Target number of frames: {rounded_frames}")

        # keep the last frames
        keep_indices = np.arange(total_frames - rounded_frames, total_frames, dtype=int)

        # Write to temporary file
        frame0 = traj[int(keep_indices[0])]
        ref_frame_whole = frame0.particles.position[:].copy()
        ref_frame_whole = wholeChains(ref_frame_whole, Nchains, chainLength, frame0.configuration.box[0], frame0.configuration.box[1], frame0.configuration.box[2])

        # with gsd.hoomd.open(temp_file, 'wb') as trimmed:
        #     for i in keep_indices[0:3]:
        #         frame = traj[int(i)]
        #         if i < keep_indices[3]:
        #             frame.particles.position[:] = shift_by_reference(frame.particles.position[:], frame.particles.image[:], frame0.particles.position[:], frame0.particles.image[:], frame0.configuration.box[0], frame0.configuration.box[1], frame0.configuration.box[2])
        #             # frame.particles.image[:]  = frame.particles.image[:] - frame0.particles.image[:]
        #         trimmed.append(frame)

        with gsd.hoomd.open(file_out, 'wb') as trimmed:
            for i in keep_indices:
                frame = traj[int(i)]
                frame.particles.position[:] = unwrap_while_whole(frame.particles.position[:], frame.particles.image[:], frame0.particles.position[:], frame0.particles.image[:], ref_frame_whole, frame0.configuration.box[0], frame0.configuration.box[1], frame0.configuration.box[2])
                trimmed.append(frame)