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

#include "include/memory/types.h"
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
		Unit(std::string name, Types::Type type, const Grid &grid);
		Unit(std::string name, Types::Type type, int n_x, int n_y, int n_z);
		Unit(std::string name, Types::Type type, int n_x, int n_y);
		Unit(std::string name, Types::Type type, int n_x);
		Unit(std::string name, Unit<T> *copy);

		bool linkToPrimaryUnit(Base_Unit *base_unit, std::string &message);
		bool isLinked() { return is_secondary_linked_memory; }

		bool allocateMemory(std::string &message);
		bool deallocateMemory(std::string &message);
		bool copyDeviceToHost(std::string &message);
		bool copyHostToDevice(std::string &message);

		std::string toString();

		T* getDataHost();
		T* getDataDevice();

	private:
		T *data_host;
		T *data_device;

		Unit<T> *primary_unit;

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
