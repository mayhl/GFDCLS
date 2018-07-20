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

#ifndef MEMORY_MANAGER
#define MEMORY_MANAGER

#include "include/memory/type.h"
#include "include/memory/unit.h"

#include <string>

// -------------------------------------------------------------------- //
//                                                                      //
// Memory Manager v1.0                                                  //
//                                                                      //
// Purpose: Wrapper functions to allocate/deallocate memory on host     //
//          and/or device and copy data between for a variable number   //
//          of arguments/memory units.                                  //
//                                                                      //
// Notes: Uses variadic function form, a recursive style calling        //
//        procedure for a variable number of arguments.                 //
//                                                                      //
//                                                                      //
// Requirements: At least C++ version 11.                               //
//                                                                      //
// -------------------------------------------------------------------- //
namespace Memory
{
	namespace Manager
	{
		template<typename T, typename ...Types> 
			bool allocateMemory
				(std::string &message, Unit<T> *first, Types* ... rest);
				
		template<typename T> 
			bool allocateMemory
				(std::string &message, Unit<T> *first);
		
		
		template<typename T, typename ...Types> 
			bool deallocateMemory
				(std::string &message, Unit<T> *first, Types* ... rest);
				
		template<typename T> 
			bool deallocateMemory
				(std::string &message, Unit<T> *first);
		

		template<typename T, typename ...Types>
			bool copyDeviceToHost
				(std::string &message, Unit<T> *first, Types* ... rest);
				
		template<typename T> 
			bool copyDeviceToHost
				(std::string &message, Unit<T> *first);

		
		template<typename T, typename ...Types>
			bool copyHostToDevice
				(std::string &message, Unit<T> *first, Types* ... rest);
				
		template<typename T> 
			bool copyHostToDevice
				(std::string &message, Unit<T> *first);
	
	};

};

#include "src/memory/manager.tpp"

#endif
