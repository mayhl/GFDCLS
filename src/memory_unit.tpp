// ----------------------------------------------------------------------------------
// Copyright 2016-2018 Michael-Angelo Yick-Hang Lam
//
// The development of this software was supported by the National Science Foundation
// (NSF) Grant Number DMS-1211713.
//
// This file is part of GADIT.
//
// GADIT is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License version 3 as published by
// the Free Software Foundation.
//
// GADIT is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with GADIT.  If not, see <http://www.gnu.org/licenses/>.
// ----------------------------------------------------------------------------------

#include "../include/memory_unit.h"


//////////////////
// Constructors //
//////////////////

template<class T>
	Memory_Unit<T>::Memory_Unit(std::string name, MemoryType::Type type, int n_x, int n_y, int n_z) :
		name(name), type(type), n_x(n_x), n_y(n_y), n_z(n_z)
	{
		initialize();
	}
	
template<class T>
	Memory_Unit<T>::Memory_Unit(std::string name, MemoryType::Type type, int n_x, int n_y) :
		name(name), type(type), n_x(n_x), n_y(n_y), n_z(1)
	{
		initialize();
	}
	
template<class T>
	Memory_Unit<T>::Memory_Unit(std::string name, MemoryType::Type type, int n_x) :
		name(name), type(type), n_x(n_x), n_y(1), n_z(1)
	{
		initialize();
	}

template<class T>	
	Memory_Unit<T>::Memory_Unit(std::string name, Memory_Unit<T> *copy) :	
		name(name), type(copy->type), n_x(copy->n_x), n_y(copy->n_y), n_z(copy->n_z), memory_size(copy->memory_size)
	{
		initialize();
	}
	
	
//////////////////////////////
// Constructors Subroutines //
//////////////////////////////
	
template<class T>	
	void Memory_Unit<T>::initialize()
	{
		data_device = NULL;
		data_host = NULL;
		is_host_allocated = false;
		is_device_allocated = false;
		computeMemorySize();
	}	
	
template<class T>
	void Memory_Unit<T>::computeMemorySize()
	{
		memory_size = n_x*n_y*n_z*sizeof(T);

		dimensions = 0;
		if (n_x > 1) { dimensions++; }
		if (n_y > 1) { dimensions++; }
		if (n_z > 1) { dimensions++; }
	}

	
///////////////////////
// Memory Operations //
///////////////////////

	
template<typename T> 
	bool Memory_Unit<T>::allocateMemory(std::string &message)
	{

		cudaError cuda_error_device = cudaSuccess;
		cudaError cuda_error_host = cudaSuccess;
		bool is_device_successful = false; 
		bool is_host_successful = false; 
		bool is_successful = false; 
		
		// Allocating memory depending on type.
		switch (type)
		{
		case MemoryType::host_only:
		
			if ( !is_host_allocated )
				data_host = (T*)malloc(memory_size);		
			break;

		case MemoryType::device_only:
		
			if ( !is_device_allocated )
				cuda_error_device = cudaMalloc((void **)&data_device, memory_size);
			break;

		case MemoryType::pinned:
		
			if ( !is_host_allocated )
				cuda_error_host = cudaMallocHost((void**)&data_host, memory_size);
			
			if ( !is_device_allocated )
				cuda_error_device = cudaMalloc((void **)&data_device, memory_size);
			break;
	
		case MemoryType::non_pinned:
			
			if ( !is_host_allocated )
				data_host = (T*)malloc(memory_size);
			
			if ( !is_device_allocated )
				cuda_error_device = cudaMalloc((void **)&data_device, memory_size);	
			break;

		}
	
		////////////////////////////////
		// NOTE: Clean up logic below //
		////////////////////////////////
		
		// Verifying host memory has been allocated and detecting error type.
		std::string host_error_message= "ERROR: No Message produce.\n\n";

		if ( type == MemoryType::device_only )
			is_host_successful = true;
		else
		{
			if( is_host_allocated )
			{
				is_host_successful = false;
				host_error_message = "Memory has been previously allocated.\n\n";
			}
			else
			{
				if (type == MemoryType::pinned )
				{
					if (cuda_error_host == cudaSuccess)				
						is_host_successful = true;		
					else 
					{
						is_host_successful = false;
						host_error_message  = ("Type    : " + std::string(cudaGetErrorName(cuda_error_host)) + "\n");
						host_error_message += ("Message : " + std::string(cudaGetErrorString(cuda_error_host)) + "\n\n");
					}
				}
				else
				{				
					if ( data_host )
						is_host_successful = true;
					else
					{
						is_host_successful = false;
						host_error_message = "Memory not allocated correctly, host pointer is null.\n\n";
					}
				}
			}
		}

		// Verifying device memory has been allocated and detecting error type.
		std::string device_error_message= "ERROR: No Message produce.\n\n";
		
		if( type == MemoryType::host_only )
			is_device_successful = true;
		else
		{
			if( is_device_allocated )
			{
				is_device_successful = false;
				device_error_message = "Memory has been previously allocated.\n\n";
			}
			else
			{
				if (cuda_error_device == cudaSuccess)
					is_device_successful = true;					
				else 
				{
					is_device_successful = false;
					device_error_message  = ("Type    : " + std::string(cudaGetErrorName(cuda_error_device)) + "\n");
					device_error_message += ("Message : " + std::string(cudaGetErrorString(cuda_error_device)) + "\n\n");
				}
			}
		}
		
		// Setting flags
		if ( is_host_successful )
			is_host_allocated = true;
		
		if ( is_device_successful )
			is_device_allocated = true;
	
		
		// Constructing output message.
		if( is_device_successful && is_host_successful )
		{
			is_successful = true;
			message += "Allocating " + MemoryType::toString(type) + " memory '" + name + "' successful.\n";
			
		}
		else
		{
			is_successful = false;
			
			message += "\n------------------------\n";
			message +=   "     ALLOCATE ERROR     ";
			message += "\n------------------------\n\n";
			
			if( !is_host_successful )
			{
				message += "Host Memory Error\n";
				message += "-----------------\n";
				message += host_error_message;
			}
			
			if (!is_device_successful )
			{
				
				message += "Device Memory Error\n";
				message += "-------------------\n";
				message += device_error_message;

			}
			
			message += "Memory Unit Info\n";
			message += "----------------\n";
			message += this->toString();
			message += "\n";
		}
		
		return is_successful;
	}

template<typename T>
	bool Memory_Unit<T>::deallocateMemory(std::string &message)
	{
		
		cudaError cuda_error_device = cudaSuccess;
		cudaError cuda_error_host = cudaSuccess;
		bool is_device_successful = false; 
		bool is_host_successful = false; 
		bool is_successful = false; 
		
		switch (type)
		{
		case MemoryType::host_only:
			if (is_host_allocated )
				free(data_host);
			break;

		case MemoryType::device_only:
			if( is_device_allocated )
				cuda_error_device = cudaFree(data_device);
			break;

		case MemoryType::pinned:
			if( is_host_allocated )
				cuda_error_host = cudaFreeHost(data_host);
			
			if( is_device_allocated )
				cuda_error_device = cudaFree(data_device);
			break;

		case MemoryType::non_pinned:
			if( is_host_allocated )
				free(data_host);
			if( is_device_allocated )
				cuda_error_device = cudaFree(data_device);
			break;
		}
			
		////////////////////////////////
		// NOTE: Clean up logic below //
		////////////////////////////////
		
		// Verifying host memory has been deallocated and detecting error type.
		std::string host_error_message = "ERROR: No Message produce.\n\n";
		
		if ( type == MemoryType::device_only )
			is_host_successful = true;
		else
		{
			if ( !is_host_allocated )
			{
				is_host_successful = false;
				host_error_message = "Can not deallocate memory as it has not been previously allocated.\n\n";
									
			}
			else
			{
				if( type == MemoryType::pinned )
				{
					if( cuda_error_host == cudaSuccess )					
						is_host_successful = true;
					else
					{	
						is_host_successful = false;
						host_error_message  = ("Type    : " + std::string(cudaGetErrorName(cuda_error_host)) + "\n");
						host_error_message += ("Message : " + std::string(cudaGetErrorString(cuda_error_host)) + "\n\n");	
					}
				}
				else
				{
					if ( data_host )
						is_host_successful = true;
					else
					{
						is_host_successful = false;
						host_error_message = "Pointer is invalid when is should not be.\n\n";
					}
				}
			}
		}
		
		// Verifying device memory has been deallocated and detecting error type.	
		std::string device_error_message = "ERROR: No Message produce.\n\n";;
		
		if ( type == MemoryType::host_only )
			is_device_successful = true;
		else
		{
			if( !is_device_allocated )
			{
				is_device_successful= false;
				device_error_message = "Can not deallocate memory as it has not been previously allocated.\n\n";
			}
			else
			{
				if( cuda_error_device == cudaSuccess )
					if ( data_device )
						is_device_successful = true;
					else
					{
						is_device_successful = false;
						device_error_message = "Pointer is invalid when is should not be.\n\n";
					}
					
				else
				{
					is_device_successful = false;
					device_error_message  = ("Type    : " + std::string(cudaGetErrorName(cuda_error_device)) + "\n");
					device_error_message += ("Message : " + std::string(cudaGetErrorString(cuda_error_device)) + "\n\n");
				}
			}
		}
		
		// Setting flags
		if ( is_host_successful )
		{
			is_host_allocated = false;
			data_host = NULL;
		}
		
		if ( is_device_successful )
		{
			is_device_allocated = false;
			data_device = NULL;
		}

		
		// Constructing output message.
		if( is_device_successful && is_host_successful )
		{
			is_successful = true;
			message += "Deallocating " + MemoryType::toString(type) + " memory '" + name + "' successful.\n";
			
		}
		else
		{
			is_successful = false;
			
			message += "\n--------------------------\n";
			message +=   "     DEALLOCATE ERROR     ";
			message += "\n--------------------------\n\n";
			
			if( !is_host_successful )
			{
				message += "Host Memory Error\n";
				message += "-----------------\n";
				message += host_error_message;
			}
			
			if (!is_device_successful )
			{
				
				message += "Device Memory Error\n";
				message += "-------------------\n";
				message += device_error_message;

			}
			
			message += "Memory Unit Info\n";
			message += "----------------\n";
			message += this->toString();
			message += "\n";
		}
		
		return is_successful;

	};	
	
template<typename T>
	bool Memory_Unit<T>::copyDeviceToHost(std::string &message)
	{
		
		cudaError cuda_error = cudaSuccess;
		bool is_successful = false; 
		
		std::string submessage;
		
		if( type == MemoryType::pinned || type == MemoryType::non_pinned )
		{
			cuda_error = cudaMemcpy(data_host, data_device, memory_size, cudaMemcpyDeviceToHost);
			
			if ( cuda_error != cudaSuccess )
			{
				is_successful = false;
				submessage  = ("Type    : " + std::string(cudaGetErrorName(cuda_error)) + "\n");
				submessage += ("Message : " + std::string(cudaGetErrorString(cuda_error)) + "\n\n");
			}
			else
			{
				is_successful = true;	
				message += "Copying device to host for " + MemoryType::toString(type) + " memory '" + name + "' successful.\n";
			}
		}
		else
		{
			is_successful = false;
			submessage = "Attempting to copy data between host and device for invalid memory unit type.\n\n"; 
		}
		
		if( !is_successful)
		{
			message += "\n-------------------------------------\n";
			message +=   "     MEMCPY DEVICE TO HOST ERROR     ";
			message += "\n-------------------------------------\n\n";
			
			message += submessage;
			
			message += "Memory Unit Info\n";
			message += "----------------\n";
			message += this->toString();
			message += "\n";
		}
	
		
		return is_successful;
	}

template<typename T>
	bool Memory_Unit<T>::copyHostToDevice(std::string &message)
	{
		
		cudaError cuda_error = cudaSuccess;
		bool is_successful = false; 
		
		std::string submessage;
		
		if( type == MemoryType::pinned || type == MemoryType::non_pinned )
		{
			cuda_error = cudaMemcpy(data_device, data_host, memory_size, cudaMemcpyHostToDevice);
			
			if ( cuda_error != cudaSuccess )
			{
				is_successful = false;
				submessage  = ("Type    : " + std::string(cudaGetErrorName(cuda_error)) + "\n");
				submessage += ("Message : " + std::string(cudaGetErrorString(cuda_error)) + "\n\n");
			}
			else
			{
				is_successful = true;	
				message += "Copying host to device for " + MemoryType::toString(type) + " memory '" + name + "' successful.\n";
			}
		}
		else
		{
			is_successful = false;
			submessage = "Attempting to copy data between host and device for invalid memory unit type.\n\n"; 
		}
		
		if( !is_successful)
		{
			message += "\n-------------------------------------\n";
			message +=   "     MEMCPY HOST TO DEVICE ERROR     ";
			message += "\n-------------------------------------\n\n";
			
			message += submessage;
			
			message += "Memory Unit Info\n";
			message += "----------------\n";
			message += this->toString();
			message += "\n";
		}
	
		
		return is_successful;
		
	}	

	
/////////////////
// Memory Info //
/////////////////		
	
template<typename T>
	std::string Memory_Unit<T>::toString()
	{
		std::string message = "";
		
		message += ("Name        : " + name + "\n");
		message += ("Memory Type : " + MemoryType::toString(type) + "\n");
		message += ("Data Type   : " + std::string(typeid(T).name()) + "\n");
		message += ("Dims        : " + std::to_string(static_cast<unsigned long long>(dimensions)) + "\n");
		message += ("n_x         : " + std::to_string(static_cast<unsigned long long>(n_x)) + "\n");
		message += ("n_y         : " + std::to_string(static_cast<unsigned long long>(n_z)) + "\n");
		message += ("n_z         : " + std::to_string(static_cast<unsigned long long>(n_z)) + "\n"); 
		message += ("Memory Size : " + std::to_string(static_cast<unsigned long long>(memory_size)) + "\n");
		
		return message;
	}
	
template<class T>	
	MemoryType::Type Memory_Unit<T>::getType()
	{
		return type;
	}
	
template<class T>
	size_t Memory_Unit<T>::getMemorySize()
	{
		return memory_size;
	}
	
template<class T>
	size_t Memory_Unit<T>::getSize_x()
	{
		return n_x;
	}

template<class T>
	size_t Memory_Unit<T>::getSize_y()
	{
		return n_y;
	}
	
template<class T>
	size_t Memory_Unit<T>::getSize_z()
	{
		return n_z;
	}