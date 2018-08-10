// -------------------------------------------------------------------- //
// Copyright 2016-2018 Michael-Angelo Yick-Hang Lam                     //
//                                                                      //
// The development of this software was supported by the National       //
// Science Foundation (NSF) Grant Number DMS-1211713.                   //
//                                                                      //
// This file is part of GADIT.                                          //
//                                                                      //
// GADIT is free software: you can redistribute it and/or modify it     //
// under the terms of the GNU General Public License version 3 as       //
// published by the Free Software Foundation.                           //
//                                                                      //
// GADIT is distributed in the hope that it will be useful, but WITHOUT //
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY   //
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public     //
// License for more details.                                            //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with GADIT.  If not, see <http://www.gnu.org/licenses/>.       //
// ---------------------------------------------------------------------//

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
			Base_Unit(std::string name, Types::Type type, dim3 dimensions);
			Base_Unit(std::string name, Types::Type type, size_t n_x , size_t n_y , size_t n_z );

			virtual bool linkToPrimaryUnit(Base_Unit *base_unit, std::string &message) = 0;

			virtual bool allocateMemory  (std::string &message) = 0;
			virtual bool deallocateMemory(std::string &message) = 0;
			virtual bool copyDeviceToHost(std::string &message) = 0;
			virtual bool copyHostToDevice(std::string &message) = 0;

			bool compareDimensions(Base_Unit *other, std::string &message);
			std::string getName();

			size_t getMemorySize();

		protected:

			Types::Type getType();

			size_t getSizeX();
			size_t getSizeY();
			size_t getSizeZ();

			std::string name;
			Types::Type type;
			size_t memory_size;

			size_t n_x;
			size_t n_y;
			size_t n_z;

			unsigned short int dimensions;
	};
};

#endif