---
layout: default
title: Learning Continuum-level Closures For Control of Interacting Active Particles
description: Titus Quah<sup>a</sup>, Sho C. Takatori<sup>b</sup>, and James B. Rawlings<sup>a</sup><br><sup>a</sup>Department of Chemical Engineering, University of California, Santa Barbara<br><sup>b</sup>Department of Chemical Engineering, Stanford University
theme: jekyll-theme-cayman
---
<div class="resources-section">
  <style>
    .resources-section {
      background: #f8f9fa;
      padding: 2rem;
      border-radius: 8px;
      margin: 2rem 0;
    }
    
    .resource-buttons {
      display: flex;
      gap: 1rem;
      flex-wrap: wrap;
      justify-content: center;
    }
    
    .resource-link {
      display: inline-flex;
      align-items: center;
      padding: 0.8rem 1.5rem;
      border-radius: 6px;
      text-decoration: none;
      font-weight: 600;
      transition: transform 0.2s, box-shadow 0.2s;
    }
    
    .resource-link:hover {
      transform: translateY(-2px);
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .github-link {
      background: #24292e;
      color: white;
    }
    
    .arxiv-link {
      background: #b31b1b;
      color: white;
    }
    
    .resource-link img {
      width: 24px;
      height: 24px;
      margin-right: 8px;
    }
  </style>

  <div class="resource-buttons">
    <a href="https://github.com/titusswsquah/graybox_abp_mpc" class="resource-link github-link">
      <svg height="24" width="24" viewBox="0 0 16 16" fill="currentColor">
        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/>
      </svg>
      View on GitHub
    </a>
    
    <a href="https://arxiv.org/abs/2501.18809" class="resource-link arxiv-link">
      <svg height="24" width="24" viewBox="0 0 24 24" fill="currentColor">
        <path d="M19 3H5c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h14c1.1 0 2-.9 2-2V5c0-1.1-.9-2-2-2zm-3 14H8c-.55 0-1-.45-1-1s.45-1 1-1h8c.55 0 1 .45 1 1s-.45 1-1 1zm0-4H8c-.55 0-1-.45-1-1s.45-1 1-1h8c.55 0 1 .45 1 1s-.45 1-1 1zm0-4H8c-.55 0-1-.45-1-1s.45-1 1-1h8c.55 0 1 .45 1 1s-.45 1-1 1z"/>
      </svg>
      Read on arXiv
    </a>
  </div>
</div>
# Learning Advection Diffusion Universal Differential Equation (UDE) for Control
<div align="center">
  <img src="./media/fig1.png" alt="Overview of UDE+mpc" class="image-900x557">
  <p><i>Scientific machine learning framework for modeling and control of interacting active matter.
	(a) Flow chart of the proposed framework. We use agent-based simulations to learn closures within a continuum-level universal differential equation (UDE). This UDE is then embedded into Model Predictive Control (MPC) to enable precise control over the agent-based simulation, guiding it towards desired behaviors.
    (b) Schematic of 2D interacting active Brownian particles (ABPs) under an actuated aligning external field that orients particles left or right.
    % More generally, our framework may be applied to systems where the particle density is observable, including microscopy images of controllable active particles.
    (c) The UDE is formulated as an advection-diffusion equation. A neural operator represents the effective velocity field, learning its dependence on particle number density <span style="font-style:italic;">n</span>, external field Ω, and past densities and fields <span style="font-style:italic;">p</span>. The UDE is simulated with a differentiable solver under random external field sequences, starting from an initial density field. By comparing predicted density trajectories to those from agent-based simulations, we learn the advection term, providing closure relations for microscopic effects such as self-propulsion, interactions, and external field responses. Only macroscopic density fields, estimated from agent-based simulations, are used for training; individual particle trajectories are not directly used.
    (d) The learned UDE is embedded into MPC, enabling us to control emergent behaviors of the active matter system. MPC optimizes the external field sequence to achieve a desired goal, such as splitting the particle population into distinct groups or dynamically controlling particle fluxes. Our framework is robust, capable of handling particle interactions, and optimizes control actions for precise and adaptive manipulation of active matter.
  </i></p>
</div>
# UDE-MPC for interacting ABPs
## System: Magnetic-like actuator on Active Brownian Particles
<div align="center">
  <img src="./media/gif0.gif" alt="Demo of ABPS" class="image-900x600">
  <p><i> The particles in the BD simulation (left) are colored by their orientation as indicated by the legend. The number density (middle) corresponds to the BD simulation. The input torque (right) is the control signal for the magnetic-like actuator and starts with a step that orients particles to the center, then negative to orient particles to the left, and finally positive orient particles to the right. There are periodic boundary conditions at the walls.
	</i></p>
</div>
## Training trajectory
<div align="center">
    <img src="./media/gif1.gif" alt="Training data (active hard disks)" class="image-900x600">
  <p><i>Training trajectory for active hard disks with Peclet number=100, volume fraction of 40% and no confinement. Random orienting fields (right) are applied to the BD simulation (left). The number density trajectory (second from left) corresponds to the BD simulation and is used to train the advection diffusion UDE.
	</i></p>
</div>
## Testing trajectory
<div align="center">
    <img src="./media/gif2.gif" alt="Testing data (active hard disks)" class="image-900x1125">
  <p><i>Prediction accuracy of the advection diffusion UDE model over a 100 τ<sub>R</sub> time horizon using testing data. (a) The L<sup>2</sup> norm of the error between the predicted and measured number densities remains below 0.01 throughout the simulation, indicating a close match to the true number density profile. X-marks indicate points where the actuator field changes. (b) Snapshots of the particle simulation are shown for six time points: t=[0τ<sub>R</sub>, 20τ<sub>R</sub>, 40τ<sub>R</sub>, 60τ<sub>R</sub>, 80τ<sub>R</sub>, 99τ<sub>R</sub>]. Particles are colored based on their polar order in the x direction, normalized by density, m<sub>x</sub>(x)/n(x), where m<sub>x</sub>(x) = ∫∫ cos(θ)P(x, θ, t),dθ,dy. Positive polar order (particles oriented to the right) is shown in purple, while negative polar order (particles oriented to the left) is shown in green. Regions without strong polar order are colored white. Colored arrows above the particles are included for visual clarity. (c) Snapshots of the number density field for both measured (blue) and predicted (red) values are shown for the same six time points as in (b). (d) The advective flux due to the velocity field v<sub>x</sub>n is displayed for measured (blue) and predicted (red) values, with fluctuations attributed to particle collisions. (e) The actuating field applied during the simulation is shown for context.
	</i></p>
</div>
## UDE-MPC to split and juggle population of active hard disks
<div align="center">
    <img src="./media/gif3.gif" alt="MPC (active hard disks)" class="image-900x1125">
  <p><i>Splitting and juggling the population of ABPs. The steps are as follows: (1) Split particles into two equal groups centered at x<sub>+</sub> and x<sub>-</sub>; (2) Achieve a distribution where 30% of the particles are to the left of the origin while maintaining the populations centered at x<sub>+</sub> and x<sub>-</sub>; (3) Repeat step 2 with 70% of the particles to the left of the origin; and (4) Return to step 1. (a) The sum of the mean squared distances (MSDs) of particles to their respective set points, x<sub>+</sub> and x<sub>-</sub>, is plotted. (b) The target fraction of particles to the left of the origin (red line) and the realized fraction under Model Predictive Control (MPC) (blue line) are shown. (c) Particle simulation. Particles are colored based on their expected orientation in the x direction, averaged over the y direction: purple indicates particles oriented to the right, green indicates particles oriented to the left, and white indicates no strong orientation. Red dashed lines mark the set points x<sub>+</sub> and x<sub>-</sub>. Colored arrows above the particles guide the eye. (d) Number density field. (e) Polar order field. (f) The actuating field applied to the system, extracted from the first input in the MPC sequence, is displayed.
	</i></p>
</div>
## Fastest trap problem for active hard disks
<div align="center">
    <img src="./media/max_flux_tex.png" alt="Fastest trap illustration" class="image-450x526.5">
  <p><i>Illustration of the fastest trap problem for active hard disks. Particles within the red shaded region are considered trapped. The goal is to move the trap rightward as quickly as possible while maintaining a minimum trapped population.
	</i></p>
</div>
## UDE-MPC for the fastest trap problem
<div align="center">
    <img src="./media/gif4.gif" alt="UDE-MPC for Fastest trap" class="image-900x1125">
  <p><i>Fastest trap with n<sub>min</sub>=0.6 and w<sub>trap</sub>=4. 
    (a) Trapped fraction n<sub>trap</sub>(t) (blue) compared to the minimum fraction n<sub>min</sub> (red dashed line).
    (b) Trap velocity v<sub>trap</sub> over time.
    (c) Particle simulation snapshots with particles colored by expected orientation: green (left) and purple (right). The trap center is indicated by the red dashed line, and the trapped region is shaded in red.
    (d) Density field of the particles. 
    (e) Polar order field showing the particle orientation. 
    (f) Actuator field.
	</i></p>
</div>
## Advective flux control
<div align="center">
    <img src="./media/gif5.gif" alt="Flux control" class="image-900x540">
  <p><i>Simultaneous control of particle number density and mean flux using UDE-MPC. (a) The set point mean flux (dashed red line) and measured flux (solid blue line) show accurate tracking throughout the sinusoidal profile. (b) Particle positions and polar order at representative times, highlighting leftward (green) and rightward (purple) orientations. (c) Density profiles consistently accumulate near x<sub>±</sub>, indicating the successful maintenance of the target accumulation. (d) Flux profiles confirm precise mean flux control, with measured mean flux (solid blue line) matching the set point (dashed red line). (e) Actuator field snapshots reveal dynamic control strategies, including field modulation to achieve target fluxes, paired with localized jamming around x<sub>±</sub> to maintain density accumulation about x<sub>±</sub>. Red dashed vertical lines indicate target particle accumulation points, emphasizing the controller's ability to simultaneously regulate both number density and flux across the domain.
	</i></p>
</div>
<style>
  /* 900x544 image */
  .image-900x557 {
    width: 900px;
    height: 557px;
  }
  /* 900x600 image */
  .image-900x600 {
    width: 900px;
    height: 600px;
  }
  /* 900x450 image */
  .image-900x450 {
    width: 900px;
    height: 450px;
  }
  /* 900x563 image */
  .image-900x563 {
    width: 900px;
    height: 563px;
  }
  /* 450x526.5 image */
  .image-450x526\.5 {
    width: 450px;
    height: 526.5px;
  }
    /* 900x1125 image */
  .image-900x1125 {
    width: 900px;
    height: 1125px;
  }
    /* 900x540 image */
  .image-900x540 {
    width: 900px;
    height: 540px;
  }
  /* Responsive behavior for mobile */
  @media only screen and (max-width: 768px) {
    .image-900x557, .image-900x600, .image-900x450, .image-900x563, .image-450x526\.5, .image-900x1125,.image-900x540 {
      width: 100%;
      height: auto;
    }
  }
</style>