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
// Memory Type v1.0                                                     //
//																		//
// Purpose: Simple enumeration for the various types of the Memory Unit //
//          class.                                                      //
//                                                                      //
// Note:   Separated from Memory Unit class due to multiple definitions // 
//         of toString function during compilation. Likely due to       //
//         Memory Unit being a templated class, therefore, for each     //
//         include of Memory unit, the source code for toString         //
//         function is complied. Separating the Memory Type namespace   //
//         into header file and source code fixed issue.                //                         //
//                                                                      //
// -------------------------------------------------------------------- //

#ifndef MEMORY_TYPE
#define MEMORY_TYPE

#include <string>

namespace Memory
{
	namespace Types
	{
		enum Type 
		{
			host_only,
			device_only,
			pinned,
			non_pinned
			//TEXTURE,
		};
		
		std::string toString( Type type );
	}
	
};

#endif