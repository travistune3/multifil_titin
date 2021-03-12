#!/usr/bin/env python
# encoding: utf-8
"""
hs.py - A half-sarcomere model with multiple thick and thin filaments

Created by Dave Williams on 2009-12-31.
"""

import sys
import os
import multiprocessing as mp
import unittest
import time
import ujson as json
import numpy as np
import scipy.optimize as opt
from . import af
from . import mf
from . import ti

class hs:
    """The half-sarcomere and ways to manage it"""
    def __init__(self, lattice_spacing=None, z_line=None, poisson=None,
                actin_permissiveness=None, timestep_len=1,
                time_dependence=None, starts=None):
        """ Create the data structure that is the half-sarcomere model

        Parameters:
            lattice_spacing: the surface-to-surface distance (14.0)
            z_line: the length of the half-sarcomere (1250)
            poisson: poisson ratio obeyed when z-line changes. Signficant
                values are:
                    * 0.5 - constant volume
                    * 0.0 - constant lattice spacing, default value
                    * any negative value - auxetic lattice spacing
            actin_permissiveness: how open actin sites are to binding (1.0)
            timestep_len: how many ms per timestep (1)
            time_dependence: a dictionary to override the initial lattice
                spacing, sarcomere length, and actin permissiveness at each
                timestep. Each key may contain a list of the values, to be
                iterated over as timesteps proceed. The first entry in these
                lists will override passed initial values. The valid keys
                time_dependence can control are:
                    * "lattice_spacing"
                    * "z_line"
                    * "actin_permissiveness"
            starts: starting polymer/orientation for thin/thick filaments in
                form ((rand(0,25), ...), (rand(0,3), ...))
        Returns:
            None

        This is the organizational basis that the rest of the model, and
        classes representing the other structures therein will use when
        running. It contains the following properties:

        ## Half-sarcomere properties: these are properties that can be
        interpreted as belonging to the overall model, not to any thick or
        thin filament.

        lattice_spacing:
            the face to face lattice spacing for the whole model
        m_line:
            x axis location of the m line
        h_line:
            x axis location of the h line
        hiding_line:
            x axis location below which actin sites are hidden by actin
            overlap (crossing through the m-line from adjacent half sarc)

        ## Thick Filament Properties: each is a tuple of thick filaments
        (filament_0, filament_1, filament_2, filament_3) where each
        filament_x is giving the actual properties of that particular
        filament.

        thick_location:
            each tuple location is a list of x axis locations
        thick_crowns:
            each tuple location is a tuple of links to crown instances
        thick_link:
            each tuple location is a list consisting of three (one for each
            myosin head in the crown) of either None or a link to a thin_site
        thick_adjacent:
            each tuple location is a tuple of links to adjacent thin filaments
        thick_face:
            each tuple location is a tuple of length six, each location of
            which contains a tuple of links to myosin heads that are facing
            each surrounding thin filament
        thick_bare_zone:
            a single value, the length of each filament before the first crown
        thick_crown_spacing:
            a single value, the distance between two crowns on a single filament
        thick_k:
            a single value, the spring constant of the thick filament between
            any given pair of crowns

        ## Thin Filament Properties: arranged in the same manner as the
        thick filament properties, but for the eight thin filaments

        thin_location:
            each tuple location is a list of x axis locations
        thin_link:
            each tuple location is a list consisting of entries (one for each
            thin_site on the thin_filament) of either a None or a link to a
            thick_crown
        thin_adjacent:
            each tuple location is a tuple of links to adjacent thick filaments
        thin_face:
            each tuple location is a tuple of length three, each location of
            which contains a tuple of links to thin filament sites that are
            facing each surrounding thick filament
        thin_site_spacing:
            the axial distance from one thin filament binding site to another
        thin_k:
            a single value, the spring constant of the thin filament between
            any given pair of thin binding sites

        """
        # Versioning, to be updated when backwards incompatible changes to the
        # data structure are made, not on release of new features
        self.version = 1.2
        # Parse initial LS and Z-line
        if time_dependence is not None:
            if 'lattice_spacing' in time_dependence:
                lattice_spacing = time_dependence['lattice_spacing'][0]
            if 'z_line' in time_dependence:
                z_line = time_dependence['z_line'][0]
            # actin permissiveness is set below, after thin filament creation
        self.time_dependence = time_dependence
        # The next few lines use detection of None rather than a sensible
        # default value as a passed None is an explicit selection of default
        if lattice_spacing is None:
            lattice_spacing = 14.0
        if z_line is None:
            z_line = 1250
        if poisson is None:
            poisson = 0.0
        # Record initial values for use with poisson driven ls
        self._initial_z_line = z_line
        self._initial_lattice_spacing = lattice_spacing
        self.poisson_ratio = poisson
        # Store these values for posterity
        self.lattice_spacing = lattice_spacing
        self.z_line = z_line
        # Create the thin filaments, unlinked but oriented on creation.
        thin_orientations = ([4,0,2], [3,5,1], [4,0,2], [3,5,1],
                [3,5,1], [4,0,2], [3,5,1], [4,0,2])
        np.random.seed()
        if starts is None:
            thin_starts = [np.random.randint(25) for i in thin_orientations]
        else:
            thin_starts = starts[0]
        self._thin_starts = thin_starts
        thin_ids = range(len(thin_orientations))
        new_thin = lambda id: af.ThinFilament(self, id, thin_orientations[id],
                                              thin_starts[id])
        self.thin = tuple([new_thin(id) for id in thin_ids])
        # Determine the hiding line
        self.update_hiding_line()
        # Create the thick filaments, remembering they are arranged thus:
        # ----------------------------
        # |   Actin around myosin    |
        # |--------------------------|
        # |      a1      a3          |
        # |  a0      a2      a0      |
        # |      M0      M1          |
        # |  a4      a6      a4      |
        # |      a5      a7      a5  |
        # |          M2      M3      |
        # |      a1      a3      a1  |
        # |          a2      a0      |
        # ----------------------------
        # and that when choosing which actin face to link to which thick
        # filament face, use these face orders:
        # ----------------------------------------------------
        # | Myosin face order  |       Actin face order      |
        # |--------------------|-----------------------------|
        # |         a1         |                             |
        # |     a0      a2     |  m0      m1         m0      |
        # |         mf         |      af      OR             |
        # |     a5      a3     |                     af      |
        # |         a4         |      m2         m2      m1  |
        # ----------------------------------------------------
        if starts is None:
            thick_starts = [np.random.randint(1, 4) for i in range(4)]
        else:
            thick_starts = starts[1]
        self._thick_starts = thick_starts
        self.thick = (
                mf.ThickFilament(self, 0, (
                    self.thin[0].thin_faces[1], self.thin[1].thin_faces[2],
                    self.thin[2].thin_faces[2], self.thin[6].thin_faces[0],
                    self.thin[5].thin_faces[0], self.thin[4].thin_faces[1]),
                    thick_starts[0]),
                mf.ThickFilament(self, 1, (
                    self.thin[2].thin_faces[1], self.thin[3].thin_faces[2],
                    self.thin[0].thin_faces[2], self.thin[4].thin_faces[0],
                    self.thin[7].thin_faces[0], self.thin[6].thin_faces[1]),
                    thick_starts[1]),
                mf.ThickFilament(self, 2, (
                    self.thin[5].thin_faces[1], self.thin[6].thin_faces[2],
                    self.thin[7].thin_faces[2], self.thin[3].thin_faces[0],
                    self.thin[2].thin_faces[0], self.thin[1].thin_faces[1]),
                    thick_starts[2]),
                mf.ThickFilament(self, 3, (
                    self.thin[7].thin_faces[1], self.thin[4].thin_faces[2],
                    self.thin[5].thin_faces[2], self.thin[1].thin_faces[0],
                    self.thin[0].thin_faces[0], self.thin[3].thin_faces[1]),
                    thick_starts[3])
                )
        # Now the thin filaments need to be linked to thick filaments, use
        # the face orders from above and the following arrangement:
        # ----------------------------
        # |   Myosin around actin    |
        # |--------------------------|
        # |      m3      m2      m3  |
        # |          A1      A3      |
        # |      A0      A2          |
        # |  m1      m0      m1      |
        # |      A4      A6          |
        # |          A5      A7      |
        # |      m3      m2      m3  |
        # ----------------------------
        # The following may be hard to read, but it has been checked and
        # may be moderately trusted. CDW-20100406
        self.thin[0].set_thick_faces((self.thick[3].thick_faces[4],
            self.thick[0].thick_faces[0], self.thick[1].thick_faces[2]))
        self.thin[1].set_thick_faces((self.thick[3].thick_faces[3],
            self.thick[2].thick_faces[5], self.thick[0].thick_faces[1]))
        self.thin[2].set_thick_faces((self.thick[2].thick_faces[4],
            self.thick[1].thick_faces[0], self.thick[0].thick_faces[2]))
        self.thin[3].set_thick_faces((self.thick[2].thick_faces[3],
            self.thick[3].thick_faces[5], self.thick[1].thick_faces[1]))
        self.thin[4].set_thick_faces((self.thick[1].thick_faces[3],
            self.thick[0].thick_faces[5], self.thick[3].thick_faces[1]))
        self.thin[5].set_thick_faces((self.thick[0].thick_faces[4],
            self.thick[2].thick_faces[0], self.thick[3].thick_faces[2]))
        self.thin[6].set_thick_faces((self.thick[0].thick_faces[3],
            self.thick[1].thick_faces[5], self.thick[2].thick_faces[1]))
        self.thin[7].set_thick_faces((self.thick[1].thick_faces[4],
            self.thick[3].thick_faces[0], self.thick[2].thick_faces[2]))
        # Set the timestep for all our new cross-bridges
        self.timestep_len = timestep_len
        # Set actin_permissiveness for all our new binding sites
        if time_dependence is not None:
            if 'actin_permissiveness' in time_dependence:
                actin_permissiveness = \
                        time_dependence['actin_permissiveness'][0]
        if actin_permissiveness is None:
            actin_permissiveness = 1.0
        self.actin_permissiveness = actin_permissiveness
        # Track how long we've been running
        self.current_timestep = 0
        # Create the titin filaments
        ti_thick = lambda i, j: self.thick[i].thick_faces[j]
        ti_thin = lambda i, j: self.thin[i].thin_faces[j]
        self.titin = (
            ti.Titin(self, 0, ti_thick(0, 0), ti_thin(0, 1)),
            ti.Titin(self, 1, ti_thick(0, 1), ti_thin(1, 2)),
            ti.Titin(self, 2, ti_thick(0, 2), ti_thin(2, 2)),
            ti.Titin(self, 3, ti_thick(1, 0), ti_thin(2, 1)),
            ti.Titin(self, 4, ti_thick(1, 1), ti_thin(3, 2)),
            ti.Titin(self, 5, ti_thick(1, 2), ti_thin(0, 2)),
            ti.Titin(self, 6, ti_thick(0, 5), ti_thin(4, 1)),
            ti.Titin(self, 7, ti_thick(0, 4), ti_thin(5, 0)),
            ti.Titin(self, 8, ti_thick(0, 3), ti_thin(6, 0)),
            ti.Titin(self, 9, ti_thick(1, 5), ti_thin(6, 1)),
            ti.Titin(self, 10, ti_thick(1, 4), ti_thin(7, 0)),
            ti.Titin(self, 11, ti_thick(1, 3), ti_thin(4, 0)),
            ti.Titin(self, 12, ti_thick(2, 0), ti_thin(5, 1)),
            ti.Titin(self, 13, ti_thick(2, 1), ti_thin(6, 2)),
            ti.Titin(self, 14, ti_thick(2, 2), ti_thin(7, 2)),
            ti.Titin(self, 15, ti_thick(3, 0), ti_thin(7, 1)),
            ti.Titin(self, 16, ti_thick(3, 1), ti_thin(4, 2)),
            ti.Titin(self, 17, ti_thick(3, 2), ti_thin(5, 2)),
            ti.Titin(self, 18, ti_thick(2, 5), ti_thin(1, 1)),
            ti.Titin(self, 19, ti_thick(2, 4), ti_thin(2, 0)),
            ti.Titin(self, 20, ti_thick(2, 3), ti_thin(3, 0)),
            ti.Titin(self, 21, ti_thick(3, 5), ti_thin(3, 1)),
            ti.Titin(self, 22, ti_thick(3, 4), ti_thin(0, 0)),
            ti.Titin(self, 23, ti_thick(3, 3), ti_thin(1, 0)),
        )
        # |--------------------------------------------------|
        # |            Actin & titin around myosin           |
        # |--------------------------------------------------|
        # |           a1               a3                    |
        # |                                                  |
        # |  a0       t1      a2       t4       a0           |
        # |       t0     t2        t3      t5                |
        # |           M0               M1                    |
        # |       t6     t8        t9      t11               |
        # |  a4       t7      a6       t10      a4           |
        # |                                                  |
        # |           a5     t13       a7       t16      a5  |
        # |               t12    t14        t15    t17       |
        # |                   M2                M3           |
        # |               t18    t20        t21    t23   a1  |
        # |           a1      t19      a3       t22          |
        # |                                                  |
        # |                   a2                a0           |
        # |--------------------------------------------------|
        ## CHECK_JDP ## Link Thick filament to titin

    def to_dict(self):
        """Create a JSON compatible representation of the thick filament

        Example usage: json.dumps(sarc.to_dict(), indent=1)

        Current output includes:
            version: version of the sarcomere model
            timestep_len: the length of the timestep in ms
            current_timestep: time to get a watch
            lattice_spacing: the thick to thin distance
            z_line: the z_line location
            hiding_line: where binding sites become unavailable due to overlap
            time_dependence: how "lattice_spacing", "z_line", and
                "actin_permissiveness" can change
            last_transitions: keeps track of the last state change by thick
                filament and by crown
            thick: the structures for the thick filaments
            thin: the structures for the thin filaments
        """
        sd = self.__dict__.copy() # sarc dict
        sd.pop('_timestep_len')
        sd['timestep_len'] = self.timestep_len
        sd['current_timestep'] = self.current_timestep
        # set act_perm as mean since prop access returns values at every point
        sd['actin_permissiveness'] = np.mean(self.actin_permissiveness)
        sd['thick'] = [t.to_dict() for t in sd['thick']]
        sd['thin'] = [t.to_dict() for t in sd['thin']]
        sd['titin'] = [t.to_dict() for t in sd['titin']]
        return sd

    def from_dict(self, sd):
        """ Load values from a sarcomere dict. Values read in correspond to
        the current output documented in to_dict.
        """
        # Warn of possible version mismatches
        read, current = sd['version'], self.version
        if read != current:
            import warnings
            warnings.warn("Versioning mismatch, reading %0.1f into %0.1f."
                          %(read, current))
        # Get filaments in right orientations
        self.__init__(
            lattice_spacing=sd['_initial_lattice_spacing'],
            z_line=sd['_initial_z_line'],
            poisson=sd['poisson_ratio'],
            actin_permissiveness=sd['actin_permissiveness'],
            timestep_len=sd['timestep_len'],
            time_dependence=sd['time_dependence'],
            starts=(sd['_thin_starts'], sd['_thick_starts'])
            )
        # Local keys
        self.current_timestep = sd['current_timestep']
        self._z_line = sd['_z_line']
        self._lattice_spacing = sd['_lattice_spacing']
        self.hiding_line = sd['hiding_line']
        if 'last_transitions' in sd.keys():
            self.last_transitions = sd['last_transitions']
        # Sub-structure keys
        for data, titin in zip(sd['titin'], self.titin):
            titin.from_dict(data)
        for data, thick in zip(sd['thick'], self.thick):
            thick.from_dict(data)
        for data, thin in zip(sd['thin'], self.thin):
            thin.from_dict(data)

    def run(self, time_steps=100, callback=None, bar=True):
        """Run the model for the specified number of timesteps

        Parameters:
            time_steps: number of time steps to run the model for (100)
            callback: function to be executed after each time step to
                collect data. The callback function takes the sarcomere
                in its current state as its only argument. (Defaults to
                the axial force at the M-line if not specified.)
            bar: progress bar control,False means don't display, True
                means give us the basic progress reports, if a function
                is passed, it will be called as f(completed_steps,
                total_steps, sec_left, sec_passed, process_name).
                (Defaults to True)
        Returns:
            output: the results of the callback after each timestep
        """
        # Callback defaults to the axial force at the M-line
        if callback is None:
            callback = lambda sarc: sarc.axialforce()
        # Create a place to store callback information and note the time
        output = []
        tic = time.time()
        # Run through each timestep
        for i in range(time_steps):
            self.timestep()
            output.append(callback(self))
            # Update us on how it went
            toc = int((time.time()-tic) / (i+1) * (time_steps-i-1))
            proc_name = mp.current_process().name
            if bar == True:
                sys.stdout.write("\n" + proc_name +
                    " finished timestep %i of %i, %ih%im%is left"\
                    %(i+1, time_steps, toc/60/60, toc/60%60, toc%60))
                sys.stdout.flush()
            elif type(bar) == type(lambda x:x):
                bar(i, time_steps, toc, time.time()-tic, proc_name)
        return output

    def timestep(self, current=None):
        """Move the model one step forward in time, allowing the
        myosin heads a chance to bind and then balancing forces
        """
        # Record our passage through time
        if current is not None:
            self.current_timestep = current
        else:
            self.current_timestep += 1
        # Update bound states
        self.last_transitions = [thick.transition() for thick in self.thick]
        # Settle forces
        self.settle()

    @property
    def current_timestep(self):
        """Return the current timestep"""
        return self._current_timestep

    @current_timestep.setter
    def current_timestep(self, new_timestep):
        """Set the current timestep"""
        # Update boundary conditions
        self.update_hiding_line()
        td = self.time_dependence
        i = new_timestep
        if td is not None:
            if 'lattice_spacing' in td:
                self.lattice_spacing = td['lattice_spacing'][i]
            if 'z_line' in td:
                self.z_line = td['z_line'][i]
            if 'actin_permissiveness' in td:
                self.actin_permissiveness = td['actin_permissiveness'][i]
        self._current_timestep = i
        return

    @property
    def timestep_len(self):
        """Get the length of the time step in ms"""
        return self._timestep_len

    @timestep_len.setter
    def timestep_len(self, new_ts_len):
        """Set the length of the time step in ms"""
        self._timestep_len = new_ts_len
        [thick._set_timestep(self._timestep_len) for thick in self.thick]
        return

    @property
    def actin_permissiveness(self):
        """How active & open to binding, 0 to 1, are binding sites?"""
        return [thin.permissiveness for thin in self.thin]

    @actin_permissiveness.setter
    def actin_permissiveness(self, new_permissiveness):
        """Assign all binding sites the new permissiveness, 0 to 1"""
        for thin in self.thin:
            thin.permissiveness = new_permissiveness

    @property
    def z_line(self):
        """Axial location of the z-line, length of the half sarcomere"""
        return self._z_line

    @z_line.setter
    def z_line(self, new_z_line):
        """Set a new z-line, updating the lattice spacing at the same time"""
        self._z_line = new_z_line
        # update from time_dependece dict if 'lattice_spacing' exists, else update from poisson ratio
        if self.time_dependence is None:
            self.update_ls_from_poisson_ratio()
        else:
            if 'lattice_spacing' in self.time_dependence:
                pass
                # already updated, next statement overwrites if called
            else:
                self.update_ls_from_poisson_ratio()

    @property
    def lattice_spacing(self):
        """Return the current lattice spacing"""
        return self._lattice_spacing

    @lattice_spacing.setter
    def lattice_spacing(self, new_lattice_spacing):
        """Assign a new lattice spacing"""
        self._lattice_spacing = new_lattice_spacing

    @staticmethod
    def ls_to_d10(face_dist):
        """Convert face-to-face lattice spacing to d10 spacing.

        Governing equations:
            ls = ftf, the face to face distance
            filcenter_dist = face_dist + .5 * dia_actin + .5 * dia_myosin
            d10 = 1.5 * filcenter_dist
        Values:
            dia_actin: 9nm [1]_
            dia_myosin: 16nm [2]_
            example d10: 37nm for cardiac muscle at 2.2um [3]_
        References:
            .. [1] Egelman 1985, The structure of F-actin.
                   J Muscle Res Cell Motil, Pg 130, values from 9 to 10 nm
            .. [2] Woodhead et al. 2005, Atomic model of a myosin filament in
                   the relaxed state. Nature, Pg 1195, in tarantula filament
            .. [3] Millman 1998, The filament lattice of striated muscle.
                   Physiol Rev,  Pg 375
        Note: Arguably this should be moved to a support class as it really
        isn't something the half-sarcomere knows about or does. I'm leaving it
        here as a convenience for now.

        Parameters:
            face_dist: face to face lattice spacing in nm
        Returns:
            d10: d10 spacing in nm
        """
        filcenter_dist = face_dist + 0.5 * 9 + 0.5 * 16
        d10 = 1.5* filcenter_dist # 3/2 for vert, sqrt(3) invert flight
        return d10

    @staticmethod
    def d10_to_ls(d10):
        """Convert d10 spacing to face-to-face lattice spacing

        Governing equations: See ls_to_d10
        Values: See ls_to_d10

        Parameters:
            d10: d10 spacing in nm
        Returns:
            face_dist: face to face lattice spacing in nm
        """
        filcenter_dist = d10 * 2/3 # 2/3 for vertabrate, 1/sqrt(3) invert flight
        face_dist = filcenter_dist - 0.5 * 9 - 0.5 * 16
        return face_dist

    def axialforce(self):
        """Sum of each thick filament's axial force on the M-line """
        return sum([thick.effective_axial_force() for thick in self.thick])

    def radialtension(self):
        """The sum of the thick filaments' radial tensions"""
        return sum([t.radialtension() for t in self.thick])

    def radialforce(self):
        """The sum of the thick filaments' radial forces, as a (y,z) vector"""
        return np.sum([t.radial_force_of_filament() for t in self.thick], 0)# + ...
        #sum([titin.radialforce() for titin in self.titin]) #CHECK

    def _single_settle(self):
        """Settle down now, just a little bit"""
        thick = [thick.settle() for thick in self.thick]
        thin = [thin.settle() for thin in self.thin]
        return np.max((np.max(np.abs(thick)), np.max(np.abs(thin))))

    def settle(self):
        """Jiggle those locations around until the residual forces are low

        We choose the convergence limit so that 95% of thermal forcing events
        result in a deformation that produces more axial force than the
        convergence value, 0.12pN.
        """
        converge_limit=0.12 # see doc string
        converge = self._single_settle()
        while converge>converge_limit:
            converge = self._single_settle()

    def _get_residual(self):
        """Get the residual force at every point in the half-sarcomere"""
        thick_f = np.hstack([t.axialforce() for t in self.thick])
        thin_f = np.hstack([t.axialforce() for t in self.thin])
        mash = np.hstack([thick_f, thin_f])
        return mash

    def length_perturbation(self, dist=None, n=None):
        """Get the force response of the half-sarcomere to a length perturbation"""
        if dist is None:
            dist = 0.2; #take length perturbation steps of 'dist' nm
        if n is None:
            n = 10; #take 'n' length perturbation steps.
        response = [];
        stiffness = [];
        initial_force = self.axialforce()
        initial_zline = self.z_line
        for i in range(n):
            self.z_line += dist
            self.settle()
            response.append(((i+1) * dist, self.axialforce() - initial_force))
            stiffness.append((self.axialforce() - initial_force) / ((i+1) * dist))
        self.z_line = initial_zline
        self.settle()
        return np.mean(stiffness)

    def get_frac_in_states(self):
        """Calculate the fraction of cross-bridges in each state"""
        nested = [t.get_states() for t in self.thick]
        xb_states = [xb for fil in nested for face in fil for xb in face]
        num_in_state = [xb_states.count(state) for state in range(3)]
        frac_in_state = [n/float(len(xb_states)) for n in num_in_state]
        return frac_in_state
    
    
    def get_num_in_states(self):
        """Calculate the number of cross-bridges in each state"""
        nested = [t.get_states() for t in self.thick]
        xb_states = [xb for fil in nested for face in fil for xb in face]
        num_in_state = [xb_states.count(state) for state in range(3)]
        return num_in_state

    #ADDED BY JDP ON 2017-Aug-21
    def get_31_trans(self):
        """Calculate the number of xb's that transition from state 3 to 1"""
        xb_trans = sum(sum(self.last_transitions,[]),[])
        return xb_trans.count('31')

    def update_ls_from_poisson_ratio(self):
        """Update the lattice spacing consistant with the poisson ratio,
        initial lattice spacing, current z-line, and initial z-line

        Governing equations
        ===================
        Poisson ratio := ν
            ν = dε_r/dε_z = Δr/r_0 / Δz/z_0
        From Mathematica derivation
        γ := center to center distance between filaments
            γ(ν, γ_0, z_0, Δz) = γ_0 (z_0/(z_0+Δz))^ν
        And since we want the face-to-face distance, aka ls, we convert with:
            γ = ls + 0.5 (dia_actin + dia_myosin)
        and
            γ_0 = ls_0 + 0.5 (dia_actin + dia_myosin)
        and the simplifying
            β = 0.5 (dia_actin + dia_myosin)
        to get
            ls = (ls_0 + β) (z_0/(z_0 + Δz))^ν - β
        which is what we implement below.
        Note: this is a novel derivation and so there is no current
            citation to be invoked.

        Values: See ls_to_d10

        Parameters:
            None
        Returns:
            None
        """
        beta =  0.5 * (9 + 16)
        ls_0 = self._initial_lattice_spacing
        z_0 = self._initial_z_line
        nu = self.poisson_ratio
        dz = self.z_line - z_0
        
                
        if self.time_dependence is not None:
            # pdb.set_trace()
            if 'z_line' in self.time_dependence:
                z_0 = np.mean(self.time_dependence['z_line']) # mean will be z0 if sinusoid
        else:
            z_0 = self._initial_z_line
        
        # ls = (ls_0 + beta) * (z_0/(z_0 + dz))**nu - beta
        # self.lattice_spacing = ls
        
        # update with poisson ratio for invertebrate flight muscle
        d10_0 =  np.sqrt(3) * (ls_0 + beta) # sqrt 3 is for invert flight muscle, 3/2 would be for vert 
        delta_d10 = - d10_0 * (1 - (1 + (dz)/z_0 )**(-nu) )
        ls = (d10_0 + delta_d10)/np.sqrt(3) - beta
        self.lattice_spacing = ls
        
        #  update with poisson ratio for vertebrate
        # d10_0 =  3/2 * (ls_0 + beta) # sqrt 3 is for invert flight muscle, 3/2 would be for vert 
        # delta_d10 = - d10_0 * (1 - (1 + (dz)/z_0 )**(-nu) )
        # ls = (d10_0 + delta_d10)/(3/2) - beta
        # self.lattice_spacing = ls
        
        return

    def update_hiding_line(self):
        """Update the line determining which actin sites are unavailable"""
        farthest_actin = min([min(thin.axial) for thin in self.thin])
        self.hiding_line = -farthest_actin

    def resolve_address(self, address):
        """Give back a link to the object specified in the address
        Addresses are formatted as the object type (string) followed by a list
        of the indices that the object occupies in each level of organization.
        Valid string values are:
            thin_fil
            thin_face
            bs
            thick_fil
            crown
            thick_face
            xb
        and an example valid address would be ('bs', 1, 14) for the binding
        site at index 14 on the thin filament at index 1.
        """
        if address[0] == 'thin_fil':
            return self.thin[address[1]]
        elif address[0] in ['thin_face', 'bs']:
            return self.thin[address[1]].resolve_address(address)
        elif address[0] == 'thick_fil':
            return self.thick[address[1]]
        elif address[0] in ['crown', 'thick_face', 'xb']:
            return self.thick[address[1]].resolve_address(address)
        import warnings
        warnings.warn("Unresolvable address: %s"%address)

    def display_axial_force_end(self):
        """ Show an end view with axial forces of face pairs

        Parameters:
            None
        Returns:
            None
        """
        # Note: The display requires the form:
        #  [[M0_A0, M0_A1, ..., M0_A5], ..., [M3_A0, ..., M3_A5]]
        forces = [[face.axialforce() for face in thick.thick_faces]
                    for thick in self.thick]
        # Display the forces
        self.display_ends(forces, "Axial force of face pairs", True)

    def display_state_end(self, states=[1,2]):
        """ Show an end view of the current state of the cross-bridges

        Parameters:
            states: List of states to count in the display, defaults
                    to [1,2] showing the number of bound cross-bridges
        Returns:
            None
        """
        # Compensate if the passed states aren't iterable
        try:
            iter(states)
        except TypeError:
            states = [states]
        # Retrieve and process cross-bridge states
        # Note: The display requires the form:
        #  [[M0_A0, M0_A1, ..., M0_A5], ..., [M3_A0, ..., M3_A5]]
        state_count = []
        for thick in self.thick:
            state_count.append([]) # Append list for this thick filament
            for face in thick.thick_faces:
                crossbridges = face.get_xb()
                # Retrieve states
                xb_states = [xb.numeric_state for xb in crossbridges]
                # Count states that match our passed states of interest
                count = sum([state in states for state in xb_states])
                state_count[-1].append(count)
        # Display the cross-bridge states
        self.display_ends(state_count, ("Cross-bridge count in state(s) "
                                        + str(states)), False)

    def display_state_side(self, states=[1,2]):
        """ Show a side view of the current state of the cross-bridges

        Parameters:
            states: List of states to count in the display, defaults
                    to [1,2] showing the number of bound cross-bridges
        Returns:
            None
        """
        # Compensate if the passed states aren't iterable
        try:
            iter(states)
        except TypeError:
            states = [states]
        # Retrieve and process cross-bridge states
        # Note: The display requires the form:
        # [[A0_0,... A0_N], [M0A0_0,... M0A0_N], ...
        #  [M0A1_0,... M0A1_N], [A1_0,... A1_N]]
        azo = lambda x: 0 if (x is None) else 1 # Actin limited to zero, one
        oddeven = 0
        vals = []
        for thick in self.thick:
            vals.append([])
            for face in thick.thick_faces:
                m_s = [xb.numeric_state for xb in face.get_xb()]
                m_s = [m in states for m in m_s]
                while len(m_s) < 40:
                    m_s.append(-1)
                a_s = [azo(bs.bound_to) for bs in face.thin_face.binding_sites]
                if oddeven == 0:
                    vals[-1].append([])
                    vals[-1][-1].append(a_s)
                    vals[-1][-1].append(m_s)
                    oddeven = 1
                elif oddeven == 1:
                    vals[-1][-1].append(m_s)
                    vals[-1][-1].append(a_s)
                    oddeven = 0
        # Display the cross-bridge states
        title = ("Cross-bridges in state(s) " + str(states))
        for fil in vals:
            for pair in fil:
                self.display_side(pair, title=title)

    def display_ends(self, graph_values, title=None, display_as_float=None):
        """ Show the state of some interaction between the filaments

        Parameters:
            graph_values: Array of values to display in the format:
                [[M0_A0, M0_A1, ..., M0_A5], ..., [M3_A0, ..., M3_A5]]
            title: Name of what is being shown (optional)
            display_as_float: Display values as floats? Tries to determine
                which type of value was passed, but can be manually set to
                True or False (optional)
        Returns:
            None

        The display is of the format:
         +-----------------------------------------------------+
         |           [AA]              [AA]                    |
         |                                                     |
         |  [AA]     0200     [AA]     0300     [AA]           |
         |                                                     |
         |      0200      0010    0100      0050               |
         |           (MM)              (MM)                    |
         |      0100      0010    0100      0010               |
         |                                                     |
         |  [AA]     0100     [AA]     0100     [AA]           |
         |                                                     |
         |           [AA]     0400     [AA]     0100     [AA]  |
         |                                                     |
         |               0200      0020    0200      0020      |
         |                    (MM)              (MM)           |
         |               0200      0010    0300      0020      |
         |                                                     |
         |           [AA]     0600     [AA]     0300     [AA]  |
         |                                                     |
         |                    [AA]              [AA]           |
         +-----------------------------------------------------+
        """
        # Functions for converting numbers to easily displayed formats
        left_float = lambda x: "%-4.1f" % x
        right_float = lambda x: "%4.1f" % x
        left_int = lambda x: "%-4i" % x
        right_int = lambda x: "%4i" % x
        if display_as_float == True:
            l = left_float
            r = right_float
        elif type(graph_values[0][0]) == int or display_as_float == False:
            l = left_int
            r = right_int
        else:
            l = left_float
            r = right_float
        # Print the title, or not
        if title is not None:
            print("  +" + title.center(53,"-") + "+")
        else:
            print("  +" + 53*"-" + "+")
        # Print the rest
        v = graph_values # Shorthand
        print(
        "  |           [AA]              [AA]                    |\n" +
        "  |                                                     |\n" +
        "  |  [AA]     %s     [AA]     %s     [AA]           |\n"
         % (l(v[0][1]), l(v[1][1])) +
        "  |                                                     |\n" +
        "  |      %s      %s    %s      %s               |\n"
         % (l(v[0][0]), r(v[0][2]), l(v[1][0]), r(v[1][2])) +
        "  |           (MM)              (MM)                    |\n" +
        "  |      %s      %s    %s      %s               |\n"
         % (l(v[0][5]), r(v[0][3]), l(v[1][5]), r(v[1][3])) +
        "  |                                                     |\n" +
        "  |  [AA]     %s     [AA]     %s     [AA]           |\n"
         % (l(v[0][4]), l(v[1][4])) +
        "  |                                                     |\n" +
        "  |           [AA]     %s     [AA]     %s     [AA]  |\n"
         % (l(v[2][1]), l(v[3][1])) +
        "  |                                                     |\n" +
        "  |               %s      %s    %s      %s      |\n"
         % (l(v[2][0]), r(v[2][2]), l(v[3][0]), r(v[3][2])) +
        "  |                    (MM)              (MM)           |\n" +
        "  |               %s      %s    %s      %s      |\n"
         % (l(v[2][5]), r(v[2][3]), l(v[3][5]), r(v[3][3])) +
        "  |                                                     |\n" +
        "  |           [AA]     %s     [AA]     %s     [AA]  |\n"
         % (l(v[2][4]), l(v[3][4])) +
        "  |                                                     |\n" +
        "  |                    [AA]              [AA]           |\n" +
        "  +-----------------------------------------------------+")
        return

    def display_side(self, graph_values, ends=(0, 0, 0), title=None,
                     labels=("A ", "M ", "A "), display_zeros=True):
        """Show the states of the filaments, as seen from their sides

        The input is essentially a list of dictionaries, each of which
        contains the values necessary to produce one of the panels this
        outputs. Each of those dictionaries contains the title (if any)
        for that panel, the side titles, the end values, and the numeric
        interaction values. Currently, the interaction values are limited
        to integers.

        Parameters:
            graph_values: Values to display in the format
                [[A0_0, A0_1, ..., A0_N],
                 [M0A0_0, M0A0_1, ..., M0A0_N],
                 [M0A1_0, M0A1_1, ..., M0A1_N],
                 [A1_0, A1_1, ..., A1_N]]
            ends: None or values for ends in the format
                [A0_end, M0_end, A1_end]
            title: None or a title string
            labels: None or filament labels in the format
                ['A0', 'M0', 'A1']
            display_zeros: Defaults to True
        Returns:
            None

        The printed output is of the format:
         +-----------------------------------------------------------+----+
         | Z-disk                                                    |    |
         | ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*         | A0 |
         | 000   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
         |                                                           |    |
         |      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00    000  |    |
         |      #==#==#==#==#==#==#==#==#==#==#==#==#==#==#======||  | M0 |
         |      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00  M-line |    |
         |                                                           |    |
         | 000   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
         | ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*         | A1 |
         | Z-disk                                                    |    |
         +-----------------------------------------------------------+----+
         |                                                           |    |
         | ||---*--*--*--*--*--*--*--*--*--*--*--*--*--*--*          | A2 |
         |                                                           |    |
         |                                                           |    |
         |         #==#==#==#==#==#==#==#==#==#==#==#==#==#==#===||  | M1 |
         |                                                           |    |
         |                                                           |    |
         | ||---*--*--*--*--*--*--*--*--*--*--*--*--*--*--*          | A3 |
         |                                                           |    |
         +-----------------------------------------------------------+----+
        ... and so on.
        """
        # Functions for converting numbers to easily displayed formats
        filter_zeros = lambda x: x if (display_zeros or (x != 0)) else None
        l = lambda x: "%-2i" % filter_zeros(int(x))
        bl = lambda x: "%-3i" % filter_zeros(int(x))
        r = lambda x: "%2i" % filter_zeros(int(x))
        br = lambda x: "%3i" % filter_zeros(int(x))
        # Print the title, if any
        if title is not None:
            print("  +" + title.center(134,"-") + "+----+")
        else:
            print("  +" + 134*"-" + "+----+")
        # Print the rest
        vals = [[bl(ends[0])] + list(map(l, graph_values[0])),
                list(map(l, graph_values[1])) + [br(ends[1])],
                list(map(l, graph_values[2])),
                [bl(ends[2])] + list(map(l, graph_values[3]))] # Shorthand
        print(
        "  | Z-disk                                                                                                                               |    |\n" +
        "  | ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*                                       | %s |\n"
            % labels[0] +
        "  | %s   %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s                                      |    |\n"
            % tuple(vals[0]) +
        "  |                                                                                                                                      |    |\n" +
        "  |      %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s    %s  |    |\n"
            % tuple(vals[1]) +
        "  |      #==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==#======||  | %s |\n"
            % labels[1] +
        "  |      %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s  M-line |    |\n"
            % tuple(vals[2]) +
        "  |                                                                                                                                      |    |\n" +
        "  | %s   %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s %s                                      |    |\n"
            % tuple(vals[3]) +
        "  | ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*                                       | %s |\n"
            % labels[2] +
        "  | Z-disk                                                                                                                               |    |\n" +
        "  +--------------------------------------------------------------------------------------------------------------------------------------+----+\n"
        )
        #+-----------------------------------------------------------+----+
        #| Z-disk                                                    |    |
        #| ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--...    | A0 |
        #| 000   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
        #|                                                           |    |
        #|      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00         |    |
        #|      #==#==#==#==#==#==#==#==#==#==#==#==#==#==#==...     | M0 |
        #|      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00         |    |
        #|                                                           |    |
        #| 000   00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
        #| ||----*--*--*--*--*--*--*--*--*--*--*--*--*--*--*--...    | A1 |
        #| Z-disk                                                    |    |
        #+-----------------------------------------------------------+----+
        #|                                                           |    |
        #|  ...--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*         | A0 |
        #|       00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
        #|                                                           |    |
        #|      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00         |    |
        #|  ...=#==#==#==#==#==#==#==#==#==#==#==#==#==#==#==...     | M0 |
        #|      00 00 00 00 00 00 00 00 00 00 00 00 00 00 00         |    |
        #|                                                           |    |
        #|       00 00 00 00 00 00 00 00 00 00 00 00 00 00 00        |    |
        #|  ...--*--*--*--*--*--*--*--*--*--*--*--*--*--*--*         | A1 |
        #|                                                           |    |
        #+-----------------------------------------------------------+----+
        #|                                                           |    |
        #|                                                           | A0 |
        #|                                                           |    |
        #|                                                           |    |
        #|      00 00 00 00 00 00 00 00 00 00  000                   |    |
        #|  ...=#==#==#==#==#==#==#==#==#==#======||                 | M0 |
        #|      00 00 00 00 00 00 00 00 00 00  M-line                |    |
        #|                                                           |    |
        #|                                                           |    |
        #|                                                           | A1 |
        #|                                                           |    |
        #+-----------------------------------------------------------+----+
        #
        #
        return


sarc = hs()
