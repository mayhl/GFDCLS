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

#ifndef MEMORY_UNIT_H
#define MEMORY_UNIT_H

#include "include/memory/type.h"
#include "include/memory/base_unit.h"

#include "include/string_processing.h"

#include <stdlib.h> 
#include "cuda_runtime.h"
#include <string>
#include <typeinfo>

namespace Memory
{
	template<class T> class Unit : public Base_Unit
	{

	public:
		T *data_host;
		T *data_device;

		Unit(std::string name, Types::Type type, size_t n_x, size_t n_y, size_t n_z);
		Unit(std::string name, Types::Type type, dim3 dimensions );
		Unit(std::string name, Unit<T> *copy);
		
		bool allocateMemory(std::string &message);
		bool deallocateMemory(std::string &message);
		bool copyDeviceToHost(std::string &message);
		bool copyHostToDevice(std::string &message);

		std::string toString();


	private:

		bool allocateNonPinnedHostMemory( std::string &message );
		bool allocatePinnedHostMemory(std::string &message);
		bool allocateDeviceMemory(std::string &message);

		bool deallocateNonPinnedHostMemory(std::string &message);
		bool deallocatePinnedHostMemory(std::string &message);
		bool deallocateDeviceMemory(std::string &message);

		void initialize();
		bool is_host_allocated;
		bool is_device_allocated;
		
		void computeMemorySize();	

	};


	// Templated classes can not be seperated into header
	// files and source files, therefore, source file
	// is include here.

}

#include "src/memory/unit.tpp"

#endif
