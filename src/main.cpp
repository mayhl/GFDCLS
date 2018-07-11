
// ----------------------------------------------------------------------------------
// Copyright 2016-2017 Michael-Angelo Yick-Hang Lam
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

// ----------------------------------------------------------------------------------
// Name:			main.cu
// Version: 		1.0
// Purpose:			Minimal example code of how to execute GADIT.
// ----------------------------------------------------------------------------------

//#include "gadit_solver.h"

#include "../include/memory_unit.h"
#include "../include/memory_manager.h"


int main()
{

	memory_unit<double> *test1, *test2, *test3, *test4, *test5;
	
	test1 = new memory_unit<double>("test1",memory_scope::PINNED, 100);
	test2 = new memory_unit<double>("test2",memory_scope::DEVICE_ONLY, 100, 100);
	test3 = new memory_unit<double>("test3",memory_scope::HOST_ONLY, 100, 100, 100);
	test4 = new memory_unit<double>("test4",memory_scope::NON_PINNED, 100, 100);
	test5 = new memory_unit<double>("test5",test4);
	
	memory_manager::allocateMemory(test1,test2,test3,test4,test5);
	memory_manager::copyDeviceToHost(test1,test2,test3,test4,test5);
	memory_manager::copyHostToDevice(test1,test2,test3,test4,test5);
	memory_manager::deallocateMemory(test1,test2,test3,test4,test5);
	
	return 0;

}


