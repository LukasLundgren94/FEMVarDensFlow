#include "hdf5.h"
#include <iostream>

int main() {
  // Open the file
  hid_t file_id = H5Fopen("mesh_reference0.h5", H5F_ACC_RDONLY, H5P_DEFAULT);

  // Retrieve the file driver handle
  void *driver_handle_void;
  herr_t status = H5Fget_vfd_handle(file_id,H5P_DEFAULT, &driver_handle_void);
  if (status < 0) {
    // Error retrieving the file driver handle
    std::cerr << "Error retrieving file driver handle" << std::endl;
    return 1;
  }
  std::cout << "status: " << status << std::endl;
  hid_t driver_handle = (hid_t) driver_handle_void;

  MPI_Info infoF;
  int ierr = MPI_File_get_info(driver_handle, &infoF);


  // Retrieve the name of the file driver
  ssize_t driver_name_size = H5Iget_name(driver_handle, NULL, 0);
  if (driver_name_size < 0) {
    // Error retrieving the size of the driver name
    std::cerr << "Error retrieving size of driver name: " << driver_name_size << std::endl;
    return 1;
  }/**/
 /* int driver_name_size = 1000;
  char *driver_name = new char[driver_name_size+1];
  ssize_t driver_name_size2 = H5Iget_name(driver_handle, driver_name, driver_name_size+1);
  if (driver_name_size2 < 0) {
    // Error retrieving the driver name
    std::cerr << "Error retrieving driver name" << std::endl;
    return 1;
  }

  // Print the name of the file driver
  std::cout << "File driver: " << driver_name << std::endl;

  delete [] driver_name;*/

  // Close the file
  H5Fclose(file_id);

  return 0;
}