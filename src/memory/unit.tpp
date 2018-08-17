//////////////////////////////////////////////////////////////////////////////////////
//                                                                                  //
//  BSD 2 - Clause License                                                          //
//                                                                                  //
//  Copyright(c) 2018, Michael-Angelo Yick-Hang Lam                                 //
//  All rights reserved.                                                            //
//                                                                                  //
//  Redistribution and use in source and binary forms, with or without              //
//  modification, are permitted provided that the following conditions are met :    //
//                                                                                  //
//  * Redistributions of source code must retain the above copyright notice, this   //
//    list of conditions and the following disclaimer.                              //
//                                                                                  //
//  * Redistributions in binary form must reproduce the above copyright notice,     //
//    this list of conditions and the following disclaimer in the documentation     //
//    and / or other materials provided with the distribution.                      //
//                                                                                  //
//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"     //
//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE       //
//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE  //
//  DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE     //
//  FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL      //
//  DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR       //
//  SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER      //
//  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,   //
//  OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    //
//  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.            //
//                                                                                  //
//  The development of this software was supported by the National                  //
//  Science Foundation (NSF) Grant Number DMS-1211713.                              //
//                                                                                  //
//  This file is part of GPU Finite Difference Conservation Law Solver (GFDCLS).    //
//                                                                                  //
//////////////////////////////////////////////////////////////////////////////////////

//////////////////
// Constructors //
//////////////////

namespace Memory
{
	template<class T>
	Unit<T>::Unit(std::string name, Types::Type type, int n_x, int n_y, int n_z) :
		Base_Unit::Base_Unit(name, type, n_x , n_y , n_z)
	{
		initialize();
	};

	template<class T>
	Unit<T>::Unit(std::string name, Types::Type type, int n_x, int n_y) :
		Base_Unit::Base_Unit(name, type, n_x, n_y, 1)
	{
		initialize();
	};

	template<class T>
	Unit<T>::Unit(std::string name, Types::Type type, int n_x) :
		Base_Unit::Base_Unit(name, type, n_x, 1, 1)
	{
		initialize();
	};

	template<class T>
	Unit<T>::Unit(std::string name, Types::Type type, const Grid &grid) :
		Base_Unit::Base_Unit(name, type, grid )
	{
		initialize();
	};

	template<class T>
	Unit<T>::Unit(std::string name, Unit<T> *copy) :
		Base_Unit::Base_Unit(name, copy->type, copy->n_x, copy->n_y , copy->n_z) 
	{ 
		initialize(); 
	};


	//////////////////////////////
	// Constructors Subroutines //
	//////////////////////////////

	template<class T>
	void Unit<T>::initialize()
	{
		data_device = NULL;
		data_host = NULL;
		is_host_allocated = false;
		is_device_allocated = false;
		is_secondary_linked_memory = false;
		computeMemorySize();
	}

	template<class T>
	void Unit<T>::computeMemorySize()
	{
		memory_size = n_x*n_y*n_z * sizeof(T);

		dimensions = 0;
		if (n_x > 1) { dimensions++; }
		if (n_y > 1) { dimensions++; }
		if (n_z > 1) { dimensions++; }
	}

	//////////////////////////
	// Link to Primary Unit //
	//////////////////////////

	template<class T>
	bool Unit<T>::linkToPrimaryUnit(Base_Unit *base_unit, std::string &message)
	{

		primary_unit = dynamic_cast<Unit<T>*>(base_unit);

		if (primary_unit)
		{
			is_secondary_linked_memory = true;
			message = "SUCCESS: Linked secondary Memory::Unit<T> '" + name + "'";
			message += " to Memory::Unit<T> '" + base_unit->getName() + "'.\n";
		}
		else 
		{
			is_secondary_linked_memory = false;
			message = "ERROR: Failed to cast base class Memory::Base_Unit '" + base_unit->getName() + "'";
			message = " to dervied class template Memory::Unit<T>.\n";
		}

		return is_secondary_linked_memory;
	}

	/////////////
	// Getters //
	/////////////

	template<class T>
	T* Unit<T>::getDataDevice()
	{
		if (is_secondary_linked_memory)
			return primary_unit->getDataDevice();
		else
			return data_device;	
	}

	template<class T>
	T* Unit<T>::getDataHost()
	{
		if (is_secondary_linked_memory)
			return primary_unit->getDataHost();
		else
			return data_host;
	}
	////////////////////////
	// Memory Subroutines //
	////////////////////////

	// Allocate Subroutines

	template<typename T>
	bool Unit<T>::allocateMemory(std::string &message)
	{
		// Skipping allocates if unit is a secondary unit
		if (is_secondary_linked_memory)
		{
			message += "SUCCESS: Not allocating memory, ";
			message += Types::toString(type) + " memory '" + name + "'";
			message += " is a secondary unit to ";
			message	+= Types::toString(primary_unit->getType() ) + " memory '" + primary_unit->getName() + "'.\n";
			 
			return true;
		}

		bool is_device_successful = true;
		bool is_host_successful = true;

		std::string host_message = "";
		std::string device_message = "";

		// Allocating memory depending on type.
		switch (type)
		{
		case Types::host_only:
			is_host_successful = allocateNonPinnedHostMemory(host_message);
			break;

		case Types::device_only:
			is_device_successful = allocateDeviceMemory(device_message);
			break;

		case Types::pinned:
			is_host_successful = allocatePinnedHostMemory(host_message);
			is_device_successful = allocateDeviceMemory(device_message);
			break;

		case Types::non_pinned:
			is_host_successful = allocateNonPinnedHostMemory(host_message);
			is_device_successful = allocateDeviceMemory(device_message);
		}

		bool is_successful = (is_host_successful && is_device_successful);

		// Constructing return message
		if (is_successful)
			message += "SUCCESS: Allocated " + Types::toString(type) + " memory '" + name + "'.\n";
		else
		{
			// If statements used to avoid introduction of indent in blank strings
			if (!is_host_successful) StringProcessing::indentLines(host_message);
			if (!is_device_successful) StringProcessing::indentLines(device_message);

			message += "ERROR: Failure in allocating " + Types::toString(type) + " memory '" + name + "'.\n";
			message += host_message;
			message += device_message;

			std::string memory_info = toString();
			StringProcessing::indentLines(memory_info);
			message += memory_info;
		}
	
		return is_successful;

	}

	template<typename T>
	bool Unit<T>::allocateNonPinnedHostMemory(std::string &message)
	{
		if (is_host_allocated)
		{
			message = "ERROR: Host memory already allocated.\n";
			return false;
		}

		data_host = (T*)malloc(memory_size);

		if (data_host)
		{
			is_host_allocated = true;
			return true;
		}
		else
		{
			message = "ERROR: Host memory not allocated, host pointer is null.\n";
			return false;
		}
	}

	template<typename T>
	bool Unit<T>::allocatePinnedHostMemory(std::string &message)
	{
		if (is_host_allocated)
		{
			message = "ERROR: Host memory already allocated.\n";
			return false;
		}

		cudaError error = cudaMallocHost((void**)&data_host, memory_size);

		if (error == cudaSuccess)
		{
			is_host_allocated = true;
			return true;
		}
		else
		{
			std::string submessage;
			StringProcessing::cudaErrorToString(error,submessage);
			StringProcessing::indentLines(submessage);

			message = "ERROR: Host memory not allocated correctly:\n";
			message += submessage;
			return false;
		}

	}

	template<typename T>
	bool Unit<T>::allocateDeviceMemory(std::string &message)
	{
		if (is_device_allocated)
		{
			message = "ERROR: Device memory already allocated.\n";
			return false;
		}

		cudaError error = cudaMalloc((void **)&data_device, memory_size);

		if (error == cudaSuccess)
		{
			is_device_allocated = true;
			return true;
		}
		else
		{
			std::string submessage;
			StringProcessing::cudaErrorToString(error, submessage);
			StringProcessing::indentLines(submessage);

			message = "ERROR: Device memory not allocated correctly:\n";
			message += submessage;

			return false;
		}

	}

	// Deallocate Subroutines

	template<typename T>
	bool Unit<T>::deallocateMemory(std::string &message)
	{

		// Skipping deallocates if unit is a secondary unit
		if (is_secondary_linked_memory)
		{
			message += "SUCCESS: Not deallocating memory, ";
			message += Types::toString(type) + " memory '" + name + "'";
			message += " is a secondary unit to ";
			message += Types::toString(primary_unit->getType()) + " memory '" + primary_unit->getName() + "'.\n";

			return true;
		}

		std::string host_message = "";
		std::string device_message = "";
		bool is_device_successful = true;
		bool is_host_successful = true;

		// Deallocating memory depending on type.
		switch (type)
		{
		case Types::host_only:
			is_host_successful = deallocateNonPinnedHostMemory(host_message);
			break;
		case Types::device_only:
			is_device_successful = deallocateDeviceMemory(device_message);
			break;

		case Types::pinned:
			is_host_successful = deallocatePinnedHostMemory(host_message);
			is_device_successful = deallocateDeviceMemory(device_message);
			break;

		case Types::non_pinned:
			is_host_successful = deallocateNonPinnedHostMemory(host_message);
			is_device_successful = deallocateDeviceMemory(device_message);
			break;
		}

		bool is_successful = (is_host_successful && is_device_successful);

		// Constructing return message
		if (is_successful)
			message += "SUCCESS: Deallocated " + Types::toString(type) + " memory '" + name + "'.\n";
		else
		{
			// If statements used to avoid introduction of indent in blank strings
			if (!is_host_successful) StringProcessing::indentLines(host_message);
			if (!is_device_successful) StringProcessing::indentLines(device_message);

			message += "ERROR: Failure in deallocating " + Types::toString(type) + " memory '" + name + "'.\n";
			message += host_message;
			message += device_message;

			std::string memory_info = toString();
			StringProcessing::indentLines(memory_info);
			message += memory_info;
		}

		return is_successful;
	}

	template<typename T>
	bool Unit<T>::deallocateNonPinnedHostMemory(std::string &message)
	{
		if (!is_host_allocated)
		{
			message = "ERROR: Deallocating host memory that has not been previously allocated.\n";
			return false;
		}

		free(data_host);

		data_host = NULL;
		is_host_allocated = false;

		return true;
	}

	template<typename T>
	bool Unit<T>::deallocatePinnedHostMemory(std::string &message)
	{
		if (!is_host_allocated)
		{
			message = "ERROR: Deallocating host memory that has not been previously allocated.\n";
			return false;
		}

		cudaError error = cudaFreeHost(data_host);

		if (error == cudaSuccess)
		{
			is_host_allocated = false;
			data_host = NULL;
			return true;
		}
		else
		{
			std::string submessage;
			StringProcessing::cudaErrorToString(error, submessage);
			StringProcessing::indentLines(submessage);

			message = "ERROR: Host memory not deallocated:\n";
			message += submessage;
			return false;
		}

	}

	template<typename T>
	bool Unit<T>::deallocateDeviceMemory(std::string &message)
	{
		if (!is_device_allocated)
		{
			message = "ERROR: Deallocating device memory that has not been previously allocated.\n";
			return false;
		}

		cudaError error = cudaFree(data_device);

		if (error == cudaSuccess)
		{
			is_device_allocated = false;
			data_device = NULL;
			return true;
		}
		else
		{
			std::string submessage;
			StringProcessing::cudaErrorToString(error, submessage);
			StringProcessing::indentLines(submessage);

			message = "ERROR: Device memory not deallocated:\n";
			message += submessage;
			return false;
		}
	}

	// Copy Subroutines

	template<typename T>
	bool Unit<T>::copyDeviceToHost(std::string &message)
	{

		bool is_successful;
		std::string submessage;

		// Attempting to copy device to host
		if ( !Types::isCopyableType(type) )
		{
			is_successful = false;
			submessage = "ERROR: Invalid memory unit type.\n";
		}
		else
		{
			cudaError error = cudaMemcpy(data_host, data_device, memory_size, cudaMemcpyDeviceToHost);

			if (error == cudaSuccess)
				is_successful = true;
			else
			{
				is_successful = false;
				std::string cuda_error_message;
				StringProcessing::cudaErrorToString(error, cuda_error_message);
				StringProcessing::indentLines(cuda_error_message);

				submessage = "ERROR: cudaMemcpy error.\n";
				submessage += cuda_error_message;

				std::string memory_info = toString();
				StringProcessing::indentLines(memory_info);
				submessage += memory_info;
			}
		}

		// Constructing return message
		if (is_successful)
			message += "SUCCESS: Copied device to host for " + Types::toString(type) + " memory '" + name + "'.\n";
		else
		{
			StringProcessing::indentLines(submessage);

			message += "ERROR: Failed to copy device to host for " + Types::toString(type) + " memory '" + name + "'.\n";
			message += submessage;
		}

		return is_successful;

	}

	template<typename T>
	bool Unit<T>::copyHostToDevice(std::string &message)
	{


		bool is_successful;
		std::string submessage;

		// Attempting to copy host to device;
		if (!Types::isCopyableType(type))
		{
			is_successful = false;
			submessage = "ERROR: Invalid memory unit type.\n";
		}
		else
		{
			cudaError error = cudaMemcpy(data_device, data_host, memory_size, cudaMemcpyHostToDevice);

			if (error == cudaSuccess)
				is_successful = true;
			else
			{
				is_successful = false;
				std::string cuda_error_message;
				StringProcessing::cudaErrorToString(error, cuda_error_message);
				StringProcessing::indentLines(cuda_error_message);

				submessage = "ERROR: cudaMemcpy error.\n";
				submessage += cuda_error_message;

				std::string memory_info = toString();
				StringProcessing::indentLines(memory_info);
				submessage += memory_info;
			}
		}

		// Constructing return message
		if (is_successful)
			message += "SUCCESS: Copied host to device for " + Types::toString(type) + " memory '" + name + "'.\n";
		else
		{
			StringProcessing::indentLines(submessage);

			message += "ERROR: Failed to copy host to device for " + Types::toString(type) + " memory '" + name + "'.\n";
			message += submessage;
		}

		return is_successful;

	}


	/////////////////
	// Memory Info //
	/////////////////		

	template<typename T>
	std::string Unit<T>::toString()
	{
		std::string message = "";

		message += ("Name        : " + name + "\n");
		message += ("Memory Type : " + Types::toString(type) + "\n");
		message += ("Data Type   : " + std::string(typeid(T).name()) + "\n");
		message += ("Dims        : " + std::to_string(static_cast<unsigned long long>(dimensions)) + "\n");
		message += ("n_x         : " + std::to_string(static_cast<unsigned long long>(n_x)) + "\n");
		message += ("n_y         : " + std::to_string(static_cast<unsigned long long>(n_y)) + "\n");
		message += ("n_z         : " + std::to_string(static_cast<unsigned long long>(n_z)) + "\n");
		message += ("Memory Size : " + std::to_string(static_cast<unsigned long long>(memory_size)) + "\n");

		return message;
	}

}
	




//template class Memory_Unit<double>;
//template class Memory_Unit<char>;	