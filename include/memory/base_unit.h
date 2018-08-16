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
//  list of conditions and the following disclaimer.                                //
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
// Memory Base Unit v1.0                                                //
//                                                                      //
// Purpose: Template-free interface for class template Memory Unit.     // 
//                                                                      //
// -------------------------------------------------------------------- //


#ifndef MEMORY_BASE_UNIT_H
#define MEMORY_BASE_UNIT_H

#include <stdlib.h> 
#include "cuda_runtime.h"
#include <string>
#include <typeinfo>

#include "include/memory/type.h"

namespace Memory
{
	class Base_Unit
	{
		public:
			Base_Unit(std::string name, Types::Type type, int n_x , int n_y , int n_z );

			virtual bool linkToPrimaryUnit(Base_Unit *base_unit, std::string &message) = 0;
			virtual bool isLinked() = 0;
			bool is_secondary_linked_memory;

			virtual bool allocateMemory  (std::string &message) = 0;
			virtual bool deallocateMemory(std::string &message) = 0;
			virtual bool copyDeviceToHost(std::string &message) = 0;
			virtual bool copyHostToDevice(std::string &message) = 0;

			bool compareDimensions(Base_Unit *other, std::string &message);
			std::string getName();

			size_t getMemorySize();
			Types::Type getType();

		protected:


			int getSizeX();
			int getSizeY();
			int getSizeZ();

			std::string name;
			Types::Type type;
			size_t memory_size;

			int n_x;
			int n_y;
			int n_z;

			unsigned short int dimensions;
	};
};

#endif