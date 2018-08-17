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
// Memory Subroutines v1.0                                              //
//																		//
// Purpose: Various miscellaneous subroutines.                          //
//                                                                      //
// Notes: Due to the class template Memory_Unit, some functions were    //
//        not added as static members to avoid repetitive definitions   //
//        for each instantiation of the class template.                 //
//                                                                      //
// -------------------------------------------------------------------- //

#ifndef MEMORY_SUBROUTINES_H
#define MEMORY_SUBROUTINES_H

#include <string>

#include "include/memory/base_unit.h"
#include "include/memory/unit.h"

namespace Memory
{
	bool isMemoryUnitsSame(Memory::Base_Unit *base_primary, Memory::Base_Unit *base_secondary, std::string &message);
	template<typename> bool isMemoryUnitsSameSubroutine(Memory::Base_Unit *base_primary, Memory::Base_Unit *base_secondary);

};

// Note: Until more function templates are added, a seperated .tpp source file
//       was not added to avoid extraneous files.
template<typename D> bool Memory::isMemoryUnitsSameSubroutine(Memory::Base_Unit *base_primary, Memory::Base_Unit *base_secondary )
{
	Memory::Unit<D> *derived_primary;
	Memory::Unit<D> *derived_secondary;

	// Note: Dynamic cast returns null pointer if particular base class  
	//       Base_Unit objects are not of derived Unit<D> type.
	derived_primary = dynamic_cast<Memory::Unit<D>*>(base_primary);
	derived_secondary = dynamic_cast<Memory::Unit<D>*>(base_secondary);

	return (derived_primary != 0) && (derived_secondary != 0);
}

#endif