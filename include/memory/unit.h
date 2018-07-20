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

#ifndef MEMORY_UNIT
#define MEMORY_UNIT

#include "include/memory/type.h"

#include <stdlib.h> 
#include "cuda_runtime.h"
#include <string>
#include <typeinfo>


template<class T> class Memory_Unit
{

public:
	T *data_host;
	T *data_device;

	
	Memory_Unit(){};
	Memory_Unit(std::string name, MemoryType::Type type, size_t n_x );
	Memory_Unit(std::string name, MemoryType::Type type, size_t n_x, size_t n_y);
	Memory_Unit(std::string name, MemoryType::Type type, size_t n_x, size_t n_y, size_t n_z);
	Memory_Unit(std::string name, Memory_Unit<T> *copy);
	


	bool allocateMemory(std::string &message);
	bool deallocateMemory(std::string &message);
	bool copyDeviceToHost(std::string &message);
	bool copyHostToDevice(std::string &message);

	MemoryType::Type getType();	
	size_t getMemorySize();
	size_t getSize_x();
	size_t getSize_y();
	size_t getSize_z();
	
	std::string toString();

private:

	std::string name;
	MemoryType::Type type;
	size_t n_x;
	size_t n_y;
	size_t n_z;
	size_t memory_size;
	unsigned short int dimensions;
	
	bool is_host_allocated;
	bool is_device_allocated;
	
//	bool is_device_clone;	
//	memory_unit<DATATYPE> *primary;

	void initialize();
	void computeMemorySize();	

};


// Templated classes can not be seperated into header
// files and source files, therefore, source file
// is include here.
#include "src/memory/unit.tpp"



#endif
