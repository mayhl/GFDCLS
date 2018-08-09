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
template<typename D> bool isMemoryUnitsSameSubroutine(Memory::Base_Unit *base_primary, Memory::Base_Unit *base_secondary )
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