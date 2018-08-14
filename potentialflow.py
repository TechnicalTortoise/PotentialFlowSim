import numpy as np
import matplotlib.pyplot as plt


class Flowspace:
    name = "flowspace"
    flow_blocks = []
    figure_count = 0
    calculated_sfvp = False
    calculated_velocity = False
    calculated_stag_points = False

    def __init__(self, xbounds, ybounds, nx, ny, talk):
        self.xbounds = xbounds
        self.ybounds = ybounds
        self.nx = nx
        self.ny = ny
        self.talk = talk
        self.stepx = (xbounds[1] - xbounds[0]) / (nx - 1)
        self.stepy = (ybounds[1] - ybounds[0]) / (ny - 1)
        self.xs = np.linspace(xbounds[0], xbounds[1], nx)
        self.ys = np.linspace(ybounds[0], ybounds[1], ny)

    #   Flow Elements =========================================================

    def add_uniform_flow(self, velocity, angle):
        """
        Adds a uniform flow to the simulation
        """
        self.flow_blocks.append((self._uniform_flow_sfvp, (velocity, angle)))

    def _uniform_flow_sfvp(self, x, y, params):
        """
        Returns the stream function and velocity potential contribution from a
        uniform flow at point (x, y), with the parameters params
        """
        velocity, angle = params
        sf = velocity * ((y * np.cos(angle)) - (x * np.sin(angle)))
        vp = velocity * ((x * np.cos(angle)) + (y * np.sin(angle)))
        return sf, vp

    def add_vortex(self, strength, x, y):
        """
        Adds a vortex to the simulation
        """
        self.flow_blocks.append((self._vortex_sfvp, (strength, x, y)))

    def _vortex_sfvp(self, x, y, params):
        """
        Returns the stream function and velocity potential contribution from a
        vortex at point (x, y), with the parameters params
        """
        strength, xpos, ypos = params
        dx = x - xpos
        dy = y - ypos
        r = ((dx ** 2) + (dy ** 2)) ** 0.5
        theta = np.arctan(dy / dx)
        sf = (0.5 * strength / np.pi) * np.log(r)
        vp = -(0.5 * strength / np.pi) * theta
        return sf, vp

    def add_source(self, strength, x, y):
        """
        Adds a source to the simulation. To simulate a sink, set strength to a
        negative value
        """
        self.flow_blocks.append((self._source_sfvp, (strength, x, y)))

    def _source_sfvp(self, x, y, params):
        """
        Returns the stream function and velocity potential contribution from a
        source at point (x, y), with the parameters params
        """
        strength, xpos, ypos = params
        dx = x - xpos
        dy = y - ypos
        r = ((dx ** 2) + (dy ** 2)) ** 0.5
        theta = np.arctan(dy / dx)
        if dx < 0 and dy > 0:
            theta += np.pi
        if dx < 0 and dy < 0:
            theta -= np.pi
        sf = (0.5 * strength / np.pi) * theta
        vp = (0.5 * strength / np.pi) * np.log(r)
        return sf, vp

    def add_doublet(self, strength, angle, x, y):
        """
        Adds a doublet to the simulation
        """
        self.flow_blocks.append((self._doublet_sfvp, (strength, angle, x, y)))

    def _doublet_sfvp(self, x, y, params):
        """
        Returns the stream function and velocity potential contribution from a
        doublet at point (x, y), with the parameters params
        """
        strength, angle, xpos, ypos = params
        dx = x - xpos
        dy = y - ypos
        r_squared = (dx ** 2) + (dy ** 2)
        sf = (-0.5 * strength / np.pi) * ((dx * np.sin(angle)) + (dy * np.cos(angle))) / r_squared
        vp = (-0.5 * strength / np.pi) * ((dy * np.sin(angle)) - (dx * np.cos(angle))) / r_squared
        return sf, vp

    #   Calculations ==========================================================

    def evaluate_sfvp(self):
        """
        Calculates the values for stream function and velocity potential at
        every point
        """
        if len(self.flow_blocks) == 0:
            print("No flow elements added")
            return
        self.sf = np.zeros((self.nx, self.ny))
        self.vp = np.zeros((self.nx, self.ny))
        for xi in range(0, self.nx):
            for yi in range(0, self.ny):
                for fb in self.flow_blocks:
                    nsf, nvp = fb[0](self.xs[xi], self.ys[yi], fb[1])
                    self.sf[xi, yi] += nsf
                    self.vp[xi, yi] += nvp
            if self.talk:
                if xi % int(self.nx / 5) == 0:
                    print("Calculating stream function + velocity potential: {}% complete".format(str(int(100 * xi / self.nx))))
                elif xi == self.nx - 1:
                    print("Calculating stream function + velocity potential: 100% complete")
        self.calculated_sfvp = True

    def evaluate_velocity(self):
        """
        Uses the values of stream function to calculate x velocity, y velocity,
        and velocity magnitude at every point
        """
        if self.calculated_sfvp is False:
            print("Error evaluating velocity, need to run evaluate_sfvp first")
            return
        self.vx = np.zeros((self.nx, self.ny))
        self.vy = np.zeros((self.nx, self.ny))
        self.vmag = np.zeros((self.nx, self.ny))
        for xi in range(0, self.nx - 1):
            for yi in range(0, self.ny - 1):
                sf0 = self.sf[xi, yi]
                sf1x = self.sf[xi + 1, yi]
                sf1y = self.sf[xi, yi + 1]
                self.vx[xi, yi] = (sf1y - sf0) / self.stepy
                self.vy[xi, yi] = -(sf1x - sf0) / self.stepx
                self.vmag[xi, yi] = ((self.vx[xi, yi]**2) + (self.vy[xi, yi]**2)) ** 0.5
            if self.talk:
                if xi % int(self.nx / 5) == 0:
                    print("Calculating velocity: {}% complete".format(str(int(100 * xi / self.nx))))
        # filling in the edges with the nearest value
        xi = self.nx - 1
        for yi in range(0, self.ny - 1):
            self.vx[xi, yi] = self.vx[xi - 1, yi]
            self.vy[xi, yi] = self.vy[xi - 1, yi]
            self.vmag[xi, yi] = self.vmag[xi - 1, yi]
        yi = self.ny - 1
        for xi in range(0, self.nx - 1):
            self.vx[xi, yi] = self.vx[xi, yi - 1]
            self.vy[xi, yi] = self.vy[xi, yi - 1]
            self.vmag[xi, yi] = self.vmag[xi, yi - 1]
        xi = self.nx - 1
        self.vx[xi, yi] = self.vx[xi - 1, yi - 1]
        self.vy[xi, yi] = self.vy[xi - 1, yi - 1]
        self.vmag[xi, yi] = self.vmag[xi - 1, yi - 1]
        # self._remove_discontinuities(self.vmag)
        if self.talk:
            print("Calculating velocity: 100% complete")
        self.calculated_velocity = True

    def evaluate_stag_points(self, tol0=10, target_n=1):
        """
        Find the positions of approximate stagnation points
        """
        if self.calculated_velocity is False:
            #print("Error evaluating stagnation points, need to run evaluate_velocity first")
            #return
            self.evaluate_velocity()
        step = 1.1
        tol = tol0
        n = target_n + 1
        while n > target_n:
            xi, yi = np.where(self.vmag < tol)
            n = len(xi)
            tol /= step
        tol *= step
        xi, yi = np.where(self.vmag < tol)
        self.stag_xs = self.xs[xi]
        self.stag_ys = self.ys[yi]
        self.stag_xi = xi
        self.stag_yi = yi
        if self.talk:
            if len(xi) > 0:
                print(str(len(xi)) + " approximate stagnation points found")
            else:
                print("No stagnation points found")
        self._evaluate_average_stag_sf()
        self.calculated_stag_points = True

    def _evaluate_average_stag_sf(self):
        """
        Evaluates the average value of stream function at the stagnation points
        found
        """
        stag_sf = np.zeros(len(self.stag_xi))
        for i in range(len(self.stag_xi)):
            stag_sf[i] = self.sf[self.stag_xi[i], self.stag_yi[i]]
        mean = np.mean(stag_sf)
        print("Average stagnation stream function is " + str(mean))
        self.stag_sf = mean

    #   Plotting ==============================================================

    def _remove_discontinuities(self, data, max_slope=40):
        """
        This function is meant to set values tending to infinity to 0.
        Currently unused
        """
        for xi in range(0, self.nx - 1):
            for yi in range(0, self.ny - 1):
                xslope = abs((data[xi + 1, yi] - data[xi, yi]) / self.stepx)
                yslope = abs((data[xi, yi + 1] - data[xi, yi]) / self.stepy)
                if xslope > max_slope:
                    data[xi, yi] = data[0, 0]
                if yslope > max_slope:
                    data[xi, yi] = data[0, 0]

    def _generate_contour_levels(self, data, n):
        """
        Creates an array of values, where for each value a contour line will be
        plotted
        """
        levels = np.linspace(np.min(data), np.max(data), n)
        return levels

    def _plot_countour(self, data, title, fill, contour_count, show_stag_point=True, show_stag_line=True, other_streamlines=()):
        X, Y = np.meshgrid(self.ys, self.xs)
        levels = self._generate_contour_levels(data, contour_count)
        plt.figure(self.figure_count)
        if fill:
            plt.contourf(Y, X, data, levels=levels)
        else:
            plt.contour(Y, X, data, levels=levels)
        plt.colorbar()
        if show_stag_point:
            if self.calculated_stag_points:
                plt.plot(self.stag_xs, self.stag_ys, 'ro')
                if self.talk:
                    print("Plotted approximate stagnation points")
            else:
                print("Error displaying stagnation points, need to run evaluate_stag_points first")
        if show_stag_line:
            if self.calculated_stag_points:
                plt.contour(Y, X, self.sf, levels=np.array([self.stag_sf]))
                if self.talk:
                    print("Plotted stagnation streamline")
            else:
                print("Error displaying stagnation streamline, need to run evaluate_stag_points first")
        if len(other_streamlines) > 0:
            for s in other_streamlines:
                plt.contour(Y, X, self.sf, levels=np.array([s]))
                if self.talk:
                    print("Plotted streamline of value " + str(s))
        plt.axes().set_aspect('equal')
        plt.title = title
        plt.xlabel("x")
        plt.ylabel("y")
        plt.show()
        self.figure_count += 1

    def plot_sf(self, fill=True, contour_count=100, show_stag_point=True, show_stag_line=True, other_streamlines=()):
        if self.calculated_sfvp:
            self._plot_countour(self.sf, "Stream Function", fill, contour_count, show_stag_point, show_stag_line, other_streamlines)
            if self.talk:
                print("Stream function plotted")
        else:
            print("Error plotting stream function, need to run evaluate_sfvp first")

    def plot_vp(self, fill=True, contour_count=100, show_stag_point=True, show_stag_line=True, other_streamlines=()):
        if self.evaluate_sfvp:
            self._plot_countour(self.vp, "Velocity Potential", fill, contour_count, show_stag_point, show_stag_line, other_streamlines)
            if self.talk:
                print("Velocity potential plotted")
        else:
            print("Error plotting velocity potential, need to run evaluate_sfvp first")

    def plot_velocitymag(self, fill=True, contour_count=100, show_stag_point=True, show_stag_line=True, other_streamlines=()):
        if self.evaluate_velocity:
            self._plot_countour(self.vmag, "Velocity Magnitude", fill, contour_count, show_stag_point, show_stag_line, other_streamlines)
            if self.talk:
                print("Velocity magnitude plotted")
        else:
            print("Error plotting velocity, need to run evaluate_velocity first")


def create_flowspace(xbounds=(-10, 10), ybounds=(-5, 5), nx=100, ny=50,
                     talk=True):
    """
    - width and height are the size of the flowspace
    - nx and ny are the number of discrete points to calculate for
    """
    fs = Flowspace(xbounds, ybounds, nx, ny, talk)
    return fs
