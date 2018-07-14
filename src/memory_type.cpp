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

#include "include/memory_type.h"

std::string MemoryType::toString( Type type )
{
	
	std::string str_type = "ERROR";
	switch(	type )
	{
		case MemoryType::host_only:
			str_type = "Host Only";
			break;
		case MemoryType::device_only:
			str_type = "Device Only";
			break;
		case MemoryType::pinned:
			str_type = "Pinned";
			break;
		case MemoryType::non_pinned:
			str_type = "Non Pinned";
			break;
	}
	
	return str_type;
};
	

