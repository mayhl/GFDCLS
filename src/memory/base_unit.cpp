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

#include "include/memory/base_unit.h"

namespace Memory
{

	Base_Unit::Base_Unit(std::string name, Types::Type type, size_t n_x, size_t n_y, size_t n_z) :
		name(name), type(type), n_x(n_x), n_y(n_y), n_z(n_z) {}

	Base_Unit::Base_Unit(std::string name, Types::Type type, dim3 dimensions ) :
		name(name), type(type), n_x(dimensions.x), n_y(dimensions.y), n_z(dimensions.z) {}

	bool Base_Unit::compareDimensions(Base_Unit *other, std::string &message)
	{
		bool is_same = false;
		if (n_x == other->getSizeX())
			if (n_y == other->getSizeY())
				if (n_z == other->getSizeZ())
					is_same = true;

		if (is_same)
			return true;
		else
		{
			message = "ERROR: Memory Units are not of the same dimensions.\n";
			return false;
		}
	}

	std::string Base_Unit::getName()
	{
		return name;
	}

	Types::Type Base_Unit::getType()
	{
		return type;
	}

	size_t Base_Unit::getMemorySize()	
	{
		return memory_size;
	}

	size_t Base_Unit::getSizeX()
	{
		return n_x;
	}

	size_t Base_Unit::getSizeY()
	{
		return n_y;
	}

	size_t Base_Unit::getSizeZ()
	{
		return n_z;
	}

}