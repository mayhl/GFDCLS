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

#include "include/memory/type.h"


namespace Memory
{
	namespace Types
	{
		std::string toString(Type type)
		{

			std::string str_type = "ERROR";
			switch (type)
			{
			case Memory::Types::host_only:
				str_type = "Host Only";
				break;
			case Memory::Types::device_only:
				str_type = "Device Only";
				break;
			case Memory::Types::pinned:
				str_type = "Pinned";
				break;
			case Memory::Types::non_pinned:
				str_type = "Non Pinned";
				break;
			}

			return str_type;
		};

		inline bool isCopyableType(Type type)
		{
			return (type == pinned || type == non_pinned);
		}
	};
};


	

