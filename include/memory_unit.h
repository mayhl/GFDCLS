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

// -------------------------------------------------------------------- //
//                                                                      //
// Memory Unit v1.0                                                     //
//																		//
// Purpose: Wrapper class to                                            //
//	           1) encapsulate pointers to GPU and CPU within a class;   //
//             2) automatically handle allocation of memory on CPU      //
//                (host) and/or GPU (device) depending on memory type:  //
//                 pinned, non-pinned, host only, or device only; and   //
//             3) simplify memory transfer between GPU and CPU when     //
//                approprite, i.e., non-pinned and pinned case.         //
//                                                                      //
// Notes: 1) Pinned memory is page locks GPU and CPU memory increasing  //
//           rate of data transfer; however,  overuse of pinned memory  //
//           may decrease performance.                                  //
//        2) This class was developed before NVIDIA Unified Memory      //
//           model was released.                                        //
//                                                                      //
//                                                                      //
// -------------------------------------------------------------------- //

#ifndef _MEMORY_UNIT_
#define _MEMORY_UNIT_

#include <stdlib.h> 
#include "cuda_runtime.h"
#include <string>
#include <typeinfo>

namespace memory_scope
{
	enum type {
		HOST_ONLY,
		DEVICE_ONLY,
		PINNED,
		NON_PINNED
		//TEXTURE,
	};
	
	std::string toString( memory_scope::type scope )
	{
		
		std::string str_scope = "ERROR";
		switch(	scope )
		{
			case memory_scope::HOST_ONLY:
				str_scope = "Host Only";
				break;
			case memory_scope::DEVICE_ONLY:
				str_scope = "Device Only";
				break;
			case memory_scope::PINNED:
				str_scope = "Pinned";
				break;
			case memory_scope::NON_PINNED:
				str_scope = "Non Pinned";
				break;
		}
		
		return str_scope;
	}
	
	
		
}

template<class DATATYPE>struct memory_unit{

public:
	DATATYPE *data_host;
	DATATYPE *data_device;

	memory_unit(std::string name, memory_scope::type scope, int n_x );
	memory_unit(std::string name, memory_scope::type scope, int n_x, int n_y);
	memory_unit(std::string name, memory_scope::type scope, int n_x, int n_y, int n_z);
	memory_unit(std::string name, memory_unit<DATATYPE> *copy);
	
	memory_scope::type getScope(void);

	bool allocateMemory(std::string &message);
	bool deallocateMemory(std::string &message);
	bool copyDeviceToHost(std::string &message);
	bool copyHostToDevice(std::string &message);
	
	size_t getMemorySize(void);
	size_t getSize_x(void);
	size_t getSize_y(void);
	size_t getSize_z(void);
	
	std::string toString(void);

private:

	std::string name;
	memory_scope::type scope;
	size_t memory_size;
	size_t n_x;
	size_t n_y;
	size_t n_z;
	unsigned short int dimensions;
	
	bool is_host_allocated;
	bool is_device_allocated;
	
//	bool is_device_clone;	
//	memory_unit<DATATYPE> *primary;

	void initialize(void);
	void computeMemorySize(void);	

};

// Templated classes can not be seperated into header
// files and source files, therefore, source file
// is include here.
#include "../src/memory_unit.tpp"

#endif
