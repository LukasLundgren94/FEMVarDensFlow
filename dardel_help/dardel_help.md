## How to use FEM_VARDEN_FLOW_Ocean

### Install fenics
It is possible to use Anaconda with legacy fenics but only works in serial (no parallel speedup).

Best option is to install legacy fenics using Spack. The following install recipe is recommended when installing fenics locally:
spack add fenics%gcc@12.3.0 cflags="-O3" fflags="-O3" openmp==False ++zlib ^python@3.10.13 ^boost@1.83.0

To install fenics on Dardel, use the install-fenics.sh script. First go to your folder inthe project folder, and create a new folder called fenics or someting there. Set the path in 
install_fenics.sh to your new folder. Run the script using sbatch, make sure to edit the project number in the script first. 

### Get the code onto Dardel

To be able to clone the repo to Dardel, you must create an ssh-key related to Dardel, that you put on your personal bitbucket page. On Dardel, do e.g.:

cd ~
ssh-keygen -t ed25519 -C "bitbucket-dardel" -f ~/.ssh/id_ed25519_bitbucket

Copy the content (the output of cat ~/.ssh/id_ed25519_bitbucket.pub) and use it to create a new key on the Bitbucket webpage (under account settings -> ssh-keys)

On dardel add the lines 

Host bitbucket.org
    IdentityFile ~/.ssh/id_ed25519_bitbucket
    IdentitiesOnly yes

to the file ~/.ssh/config 

Check that you managed to connect to bitbucket by typing ssh -T git@bitbucket.org

Create your own folder in the project folder related to the NAISS project, e.g. /cfs/klemming/projects/supr/naiss2024-6-123 (find project by typing projinfo)

Clone the repo into that folder: git clone git@bitbucket.org:Lukas_Lundgren/fem_varden_flow_old.git

Switch branch: git fetch --all
               git checkout var_den_temperature
 

### Compile model and run model


On Dardel, modify install-library.sh with your information. Then run the script from the repo root folder using sbatch.

Otherwise adapt the script to your local HPC or PC and run it.

To install one of the benchmarks run install-demo.sh inside varden/tests/benchmark. 

To run a model run the run-demo.sh script. The program reads the parameters.json file from the same directory as the run-demo.sh script.

### Awkward things

A bug in Spack installation of fenics makes XDMF output files not work properly. They have to be repaired using

tests/repair_xdmf/repair_xdmf.py *.xdmf

The same is true for the checkpoint files saved using xdmf:

tests/repair_xdmf/repair_checkpoint.py *.xdmf

Files are saved in a results/fine . Checkpoints are saved in results/checkpoint . For the results to not be overwritten, back up the results when restarting from checkpoint

mv fine fineX

### Model Setup in parameter file
For a complete list of model parameters checkout ConfigurableParameter.cpp

1. #### a) Output 
    - `86400     t_save` set time interval to output snapshots
    - `86400	t_save_avg` set time interval to output averages
    - `0       t_avg_start`: set start time to safe averages to `t0` 
    - `xdmf	    output_file_type`: Set output format to `xdmf`

1. #### b) Checkpoint restart
    - `86400.0	t_save_checkpoint` time intervall to save checkpoints
    - `864000   T` model run time

2. #### Mesh    
    - `ryder2d.xdmf     	mesh`: specifies the .xdmf file of the volume mesh (created by convert.py from .msh file)
    - `ryder2d_surface_mesh.xdmf	Surface_mesh`: specifies the .xdmf file of the surface mesh (created by convert.py from .msh file). This mesh is used to tag the boundaries to apply BC

2. #### Melt Rate parametrization
    - `1	    melting_with_rv`: Ocean heat flux based on residual viscosity:
    - `1	    melting_with_rv`Ocean heat flux based on turbulent heat exchange:
    - `3     boundary_layer_thickness` sets the distance where u_star is evaluated to `z_bl`
    - `1.5e-3	    ocean_c_d` sets the Drag coefficient (only) in the melt parametrization to `cd`

3. #### Open Ocean Boundary
    - `x_rest  restoring_region_width`: use a Sponge layer/ restoring region with `r_rest` is >0 the width of the restoring region
    - `0  restoring_region_width` use characteristic inflow/outflow and Dirichlet condition on T,S
    - for a closed boundary simply create a mesh, where the bed goes to zero where you want to close the domain

4. #### Time Stepping
    - 'time_order': 2 for Crank Nicholson, 3 for bdf3, 4 for bdf4

