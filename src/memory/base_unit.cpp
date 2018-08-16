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
//  list of conditions and the following disclaimer.                                //
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

#include "include/memory/base_unit.h"

namespace Memory
{

	Base_Unit::Base_Unit(std::string name, Types::Type type, int n_x, int n_y, int n_z) :
		name(name), type(type), n_x(n_x), n_y(n_y), n_z(n_z) {}

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

	int Base_Unit::getSizeX()
	{
		return n_x;
	}

	int Base_Unit::getSizeY()
	{
		return n_y;
	}

	int Base_Unit::getSizeZ()
	{
		return n_z;
	}

}