import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Set a seed for reproducibility
np.random.seed(42)

# Define the gravitational constant G
G = 6.67430e-11


class Particle:
    def __init__(self, mass, position):
        self.mass = mass
        self.position = np.array(position)
        self.velocity = self.generate_random_velocity()

    def generate_random_velocity(self):
        # Rejection Sampling for generating initial velocities with a normal distribution
        while True:
            v = np.random.normal(0, 1, 3)  # Mean=0, Std. Dev=1
            if np.linalg.norm(v) <= 1:
                return v


class GravitationalSimulator:
    def __init__(self, particles):
        self.particles = particles
        self.collision_prob = self.generate_collision_prob(len(particles))
    
    def evolve_with_interactions(self, t_span, dt, collision_radius):
        def system(t, y):
            num_particles = len(self.particles)
            positions = y[:3*num_particles].reshape((num_particles, 3))
            velocities = y[3*num_particles:].reshape((num_particles, 3))
            forces = np.zeros_like(positions)

            # Calculate gravitational forces
            for i in range(num_particles):
                for j in range(num_particles):
                    if i != j:
                        r = positions[j] - positions[i]
                        forces[i] += G * self.particles[i].mass * self.particles[j].mass / np.linalg.norm(r)**3 * r

            # Check for collisions
            for i in range(num_particles):
                for j in range(i+1, num_particles):
                    if i != j:
                        r = positions[j] - positions[i]
                        if np.linalg.norm(r) < collision_radius:
                            if np.random.rand() < self.collision_prob[i]:
                                # Calculate new velocities (assuming perfectly elastic collision)
                                v1 = velocities[i]
                                v2 = velocities[j]
                                m1 = self.particles[i].mass
                                m2 = self.particles[j].mass
                    
                                v1_final = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
                                v2_final = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)
                    
                                velocities[i] = v1_final
                                velocities[j] = v2_final

            return np.concatenate((velocities.flatten(), forces.flatten()))

        y0 = np.hstack([np.concatenate((p.position, p.velocity)) for p in self.particles])
        solution = solve_ivp(system, t_span, y0, method='RK45', t_eval=np.arange(t_span[0], t_span[1], dt))

        return solution.t, solution.y

    def generate_collision_prob(self, num_particles):
        # Using inverse CDF method for exponential distribution
        lambda_param = 0.5  # Can be Adjusted
        u = np.random.uniform(0, 1, num_particles)
        collision_probabilities = -np.log(1 - u) / lambda_param

        return collision_probabilities
    
def analyze_mass_distribution(t, y):
    num_particles = len(simulator.particles)
    positions = y.reshape((-1, num_particles, 6))

    # Calculate mass distribution
    mass_distribution = np.zeros((len(t), num_particles))
    for i in range(len(t)):
        for j in range(num_particles):
            mass_distribution[i, j] = np.sum([particle.mass for particle in simulator.particles])

    return mass_distribution


def plot_mass_distribution(t, mass_distribution):
    plt.plot(t, mass_distribution)
    plt.xlabel('Time')
    plt.ylabel('Total Mass')
    plt.title('Mass Distribution Over Time')
    plt.show()


def plot_3d_trajectories(t, y):
    num_particles = len(simulator.particles)
    positions = y.reshape((-1, num_particles, 6))

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, particle in enumerate(simulator.particles):
        x = positions[:, i, 0]
        y = positions[:, i, 1]
        z = positions[:, i, 2]
        ax.plot(x, y, z, label=f'Particle {i+1}')
          
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Z Position')
    ax.set_title('3D Trajectories')
    ax.legend()

    plt.show()


def plot_individual_trajectories(t, y):
    num_particles = len(simulator.particles)
    positions = y.reshape((-1, num_particles, 6))

    for i, particle in enumerate(simulator.particles):
        x = positions[:, i, 0]
        y = positions[:, i, 1]
        z = positions[:, i, 2]

        plt.figure()
        plt.plot(t, x, label='X Position')
        plt.plot(t, y, label='Y Position')
        plt.plot(t, z, label='Z Position')
        plt.xlabel('Time')
        plt.ylabel('Position')
        plt.title(f'Particle {i+1} Trajectories')
        plt.legend()
        plt.show()


def calculate_collision_probability(t, y, collision_radius):
    num_particles = len(simulator.particles)
    positions = y.reshape((-1, num_particles, 6))

    num_collisions = np.zeros(len(t))

    for i in range(len(t)):
        num_collisions_i = 0

        for j in range(num_particles):
            for k in range(j+1, num_particles):
                if np.linalg.norm(positions[i, j, :3] - positions[i, k, :3]) < collision_radius:
                    num_collisions_i += 1

        num_collisions[i] = num_collisions_i

    return num_collisions / (num_particles * (num_particles - 1) / 2)

def monte_carlo_simulation(num_simulations):
    collision_probabilities = []

    for _ in range(num_simulations):
        t_span = (0, 100)
        dt = 0.01
        t, y = simulator.evolve_with_interactions(t_span, dt, collision_radius)

        collision_probability = calculate_collision_probability(t, y, collision_radius)
        collision_probabilities.append(collision_probability)

    return np.mean(collision_probabilities, axis=0)


def calculate_total_energy(t, y):
    global simulator
    num_particles = len(simulator.particles)
    positions = y.reshape((-1, num_particles, 6))
    velocities = y.reshape((-1, num_particles, 6))

    total_energy = np.zeros(len(t))

    for i in range(len(t)):
        kinetic_energy = 0
        potential_energy = 0

        for j in range(num_particles):
            v = velocities[i, j, 3:]
            m = simulator.particles[j].mass
            kinetic_energy += 0.5 * m * np.linalg.norm(v)**2

            for k in range(j+1, num_particles):
                r = positions[i, k, :3] - positions[i, j, :3]
                m1 = simulator.particles[j].mass
                m2 = simulator.particles[k].mass
                potential_energy -= G * m1 * m2 / np.linalg.norm(r)

        total_energy[i] = kinetic_energy + potential_energy

    return total_energy


def rejection_sampling_normal(mu, sigma, num_samples):
    samples = []
    while len(samples) < num_samples:
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(0, 1)
        if y < np.exp(-x**2 / 2) / np.sqrt(2 * np.pi):
            samples.append(mu + sigma * x)
    return np.array(samples)

def inverse_cdf_exponential(lambda_param, num_samples):
    u = np.random.uniform(0, 1, num_samples)
    samples = -np.log(1 - u) / lambda_param
    return samples


def plot_collision_probability(t, collision_probabilities):
    plt.plot(t, collision_probabilities)
    plt.xlabel('Time')
    plt.ylabel('Collision Probability')
    plt.title('Collision Probability Over Time')
    plt.show()

    
def main():
    global simulator
    collision_radius = 0.1
    
    particle1 = Particle(mass=1e6, position=[5, 5, 5])
    particle2 = Particle(mass=1e6, position=[1, 1, 1])

    simulator = GravitationalSimulator([particle1, particle2])

    t_span = (0, 10)
    dt = 0.01
    global G
    G = 6.67430e-11

    t, y = simulator.evolve_with_interactions(t_span, dt, collision_radius)
    
    # Calculate collision probabilities
    collision_probabilities = calculate_collision_probability(t, y, collision_radius)
    
    # Plot collision probabilities
    plot_collision_probability(t, collision_probabilities)

    mass_distribution = analyze_mass_distribution(t, y)
    plot_mass_distribution(t, mass_distribution)

    plot_3d_trajectories(t, y)

    plot_individual_trajectories(t, y)

    collision_probability = calculate_collision_probability(t, y, collision_radius)
    print(f"Collision Probability: {collision_probability}")
    
    num_samples = 10000

    normal_samples = rejection_sampling_normal(0, 1, num_samples)
    exponential_samples = inverse_cdf_exponential(0.5, num_samples)

    plt.hist(normal_samples, bins=50, density=True, alpha=0.5, label='Rejection Sampling (Normal)')
    plt.hist(exponential_samples, bins=50, density=True, alpha=0.5, label='Inverse CDF (Exponential)')

    # Add the probability density functions for comparison
    x = np.linspace(-5, 5, 100)
    normal_pdf = np.exp(-x**2 / 2) / np.sqrt(2 * np.pi)
    exponential_pdf = 0.5 * np.exp(-0.5 * x)

    plt.plot(x, normal_pdf, label='Normal PDF')
    plt.plot(x, exponential_pdf, label='Exponential PDF')

    plt.xlabel('Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

    total_energy = calculate_total_energy(t, y)
    plt.figure()
    plt.plot(t, total_energy)
    plt.xlabel('Time')
    plt.ylabel('Total Energy')
    plt.title('Total Energy Over Time')
    plt.show()


if __name__ == "__main__":
    main()
