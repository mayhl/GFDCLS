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

#include "include/memory/subroutines.h"

namespace Memory
{
	bool isMemoryUnitsSame(Memory::Base_Unit *base_primary, Memory::Base_Unit *base_secondary, std::string &message)
	{
		// Note: If other versions of class template Unit are instantiated, subroutine
		//       needs to be modified. 
		if (isMemoryUnitsSameSubroutine<double>(base_primary, base_secondary)) return true;
		if (isMemoryUnitsSameSubroutine<char>(base_primary, base_secondary)) return true;
		if (isMemoryUnitsSameSubroutine<float>(base_primary, base_secondary)) return true;

		message = "ERROR: Memory Units are not of the same type.\n";
		return false;
	}
};