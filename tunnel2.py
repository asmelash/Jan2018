"""A wind-tunnel flow past an array of cylinders.  (20 minutes)

This example demonstrates how one can use the inlet and outlet feature of PySPH
to simulate flow inside wind tunnel. For simplicity the tunnel is periodic
along the y-axis, this is reasonable if the tunnel width is large. The fluid is
initialized with a unit velocity along the x-axis and impulsively started and
the cylinder is of unit radius. The inlet produces an incoming stream of
particles and the outlet consumes these particles. The TVF scheme is used by
default. Note that the create_equations method does some surgery of the
scheme-generated equations to adapt them for this problem.

"""
import os
import numpy as np

from pysph.base.utils import get_particle_array
from pysph.base.nnps import DomainManager
from pysph.solver.application import Application
from pysph.sph.scheme import TVFScheme
from pysph.tools import geometry as G
from pysph.sph.simple_inlet_outlet import SimpleInlet, SimpleOutlet
from pysph.sph.integrator_step import InletOutletStep

# Geometric parameters
l_tunnel = 9.0
w_tunnel = 4.0
chord = 2.0  # diameter of circle
center = l_tunnel / 3., w_tunnel / 2.  # center of circle
n_inlet = 6  # Number of inlet layers
n_outlet = 6  # Number of outlet layers

# Fluid mechanical/numerical parameters
re = 1e3
dx = 0.05
hdx = 1.2
rho = 1000
umax = 1.0
tf = 5.0

# Computed parameters
c0 = 1.5 * umax * 10
p0 = rho * c0 * c0
h0 = dx * hdx
nu = umax * chord / re
dt_cfl = 0.25 * h0 / (c0 + umax)
dt_viscous = 0.125 * h0**2 / nu
dt = min(dt_cfl, dt_viscous)


class WindTunnel(Application):
    def create_domain(self):
        i_ghost = n_inlet * dx
        o_ghost = n_outlet * dx
        domain = DomainManager(
            xmin=-i_ghost,
            xmax=l_tunnel + o_ghost,
            ymin=0,
            ymax=w_tunnel,
            periodic_in_y=True)
        return domain

    def create_particles(self):
        x, y = np.mgrid[dx:l_tunnel + dx / 4:dx, dx / 2:w_tunnel - dx / 4:dx]
        x, y = (np.ravel(t) for t in (x, y))
        one = np.ones_like(x)
        volume = dx * dx * one
        m = volume * rho
        fluid = get_particle_array(
            name='fluid',
            m=m,
            x=x,
            y=y,
            h=h0 * one,
            V=1.0 / volume,
            u=umax * one)

        xc, yc = center
        x = np.arange(xc - 0.5, xc + 0.5, dx)
        y = np.arange(yc - 0.5, yc + 0.5, dx)
        x, y = np.meshgrid(x, y)
        x, y = x.ravel(), y.ravel()
        one = np.ones_like(x)
        volume = dx * dx * one
        solid = get_particle_array(
            name='solid',
            x=x,
            y=y,
            m=volume * rho,
            rho=one * rho,
            h=h0 * one,
            V=1.0 / volume)
        G.remove_overlap_particles(fluid, solid, dx, dim=2)

        x, y = np.mgrid[dx:n_outlet * dx:dx, dx / 2:w_tunnel - dx / 4:dx]
        x, y = (np.ravel(t) for t in (x, y))
        x += l_tunnel
        one = np.ones_like(x)
        volume = dx * dx * one
        m = volume * rho
        outlet = get_particle_array(
            name='outlet',
            x=x,
            y=y,
            m=m,
            h=h0 * one,
            V=1.0 / volume,
            u=umax * one)

        # Setup the inlet particle array with just the particles we need at the
        # exit plane which is replicated by the inlet.
        y = np.arange(dx / 2, w_tunnel - dx / 4.0, dx)
        x = np.zeros_like(y)
        one = np.ones_like(x)
        volume = one * dx * dx

        inlet = get_particle_array(
            name='inlet',
            x=x,
            y=y,
            m=volume * rho,
            h=h0 * one,
            u=umax * one,
            rho=rho * one,
            V=1.0 / volume)
        self.scheme.setup_properties([fluid, inlet, outlet, solid])
        for p in fluid.properties:
            if p not in outlet.properties:
                outlet.add_property(p)

        return [fluid, solid, inlet, outlet]

    def create_scheme(self):
        s = TVFScheme(
            ['fluid', 'inlet', 'outlet'], ['solid'],
            dim=2,
            rho0=rho,
            c0=c0,
            nu=nu,
            p0=p0,
            pb=p0,
            h0=dx * hdx,
            gx=0.0)
        extra_steppers = dict(
            inlet=InletOutletStep(), outlet=InletOutletStep())
        s.configure_solver(
            extra_steppers=extra_steppers, tf=tf, dt=dt, n_damp=10, pfreq=50)
        return s

    def create_inlet_outlet(self, particle_arrays):
        f_pa = particle_arrays['fluid']
        i_pa = particle_arrays['inlet']
        o_pa = particle_arrays['outlet']

        xmin = -dx * n_inlet
        inlet = SimpleInlet(
            i_pa,
            f_pa,
            spacing=dx,
            n=n_inlet,
            axis='x',
            xmin=xmin,
            xmax=0.0,
            ymin=0.0,
            ymax=w_tunnel)
        xmax = l_tunnel + dx * n_outlet

        def callback(o_pa, new_props):
            # Set outlet particle pressure to zero.
            new_props['p'][:] = 0.0

        outlet = SimpleOutlet(
            o_pa,
            f_pa,
            xmin=l_tunnel,
            xmax=xmax,
            ymin=0.0,
            ymax=w_tunnel,
            callback=callback)
        return [inlet, outlet]

    def create_equations(self):
        eqs = self.scheme.get_equations()
        # print(eqs)  # Print the equations for your understanding.
        g0 = eqs[0]
        # Remove all the unnecessary summation density equations for the inlet
        # and outlet.
        del g0.equations[1:]
        g1 = eqs[1]
        # Remove the state equations for inlet and outlet.
        del g1.equations[1:]
        g3 = eqs[3]
        # Remove the momentum and other equations for inlet and outlet
        del g3.equations[4:]
        return eqs

    def post_process(self, info_fname):
        self.read_info(info_fname)
        if len(self.output_files) == 0:
            return

        t, fx, fy = self._plot_force_vs_t()
        res = os.path.join(self.output_dir, 'results.npz')
        np.savez(res, t=t, fx=fx, fy=fy)

    def _plot_force_vs_t(self):
        from pysph.solver.utils import iter_output, load
        from pysph.tools.sph_evaluator import SPHEvaluator
        from pysph.sph.equation import Group
        from pysph.base.kernels import QuinticSpline
        from pysph.sph.wc.transport_velocity import (
            SetWallVelocity, MomentumEquationPressureGradient,
            SolidWallNoSlipBC, SolidWallPressureBC, VolumeSummation)

        data = load(self.output_files[0])
        solid = data['arrays']['solid']
        fluid = data['arrays']['fluid']
        # We find the force of the solid on the fluid and the opposite of that
        # is the force on the solid. Note that the assumption is that the solid
        # is far from the inlet and outlet so those are ignored.
        equations = [
            Group(
                equations=[
                    VolumeSummation(
                        dest='fluid', sources=['fluid', 'solid']),
                    VolumeSummation(
                        dest='solid', sources=['fluid', 'solid']),
                ],
                real=False),
            Group(
                equations=[SetWallVelocity(
                    dest='solid', sources=['fluid']), ],
                real=False),
            Group(
                equations=[
                    SolidWallPressureBC(
                        dest='solid',
                        sources=['fluid'],
                        b=1.0,
                        rho0=rho,
                        p0=p0),
                ],
                real=False),
            Group(
                equations=[
                    # Pressure gradient terms
                    MomentumEquationPressureGradient(
                        dest='fluid', sources=['solid'], pb=p0),
                    SolidWallNoSlipBC(
                        dest='fluid', sources=['solid'], nu=nu),
                ],
                real=True),
        ]

        sph_eval = SPHEvaluator(
            arrays=[solid, fluid],
            equations=equations,
            dim=2,
            kernel=QuinticSpline(dim=2))

        t, fx, fy = [], [], []
        for sd, arrays in iter_output(self.output_files):
            fluid = arrays['fluid']
            solid = arrays['solid']
            fluid.remove_property('vmag2')
            t.append(sd['t'])
            sph_eval.update_particle_arrays([solid, fluid])
            sph_eval.evaluate()
            fx.append(np.sum(-fluid.au * fluid.m))
            fy.append(np.sum(-fluid.av * fluid.m))

        t, fx, fy = list(map(np.asarray, (t, fx, fy)))

        # Now plot the results.
        import matplotlib
        matplotlib.use('Agg')

        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(t, fx, label=r'$F_x$')
        plt.plot(t, fy, label=r'$F_y$')
        plt.xlabel(r'$t$')
        plt.ylabel('Force')
        plt.legend()
        fig = os.path.join(self.output_dir, "force_vs_t.png")
        plt.savefig(fig, dpi=300)
        plt.close()

        return t, fx, fy


if __name__ == '__main__':
    app = WindTunnel()
    app.run()
    app.post_process(app.info_filename)
