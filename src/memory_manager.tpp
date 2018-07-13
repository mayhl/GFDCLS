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


///////////////////////////////////////////////////////////
// Memory Allocate Functions                             //
///////////////////////////////////////////////////////////

template<typename T, typename ...Types>
	bool MemoryManager::allocateMemory(std::string &message, Memory_Unit<T> *first, Types* ... rest)
	{
		if ( allocateMemory(message,first) )
			return allocateMemory(message,rest...);
		else
			return false;
	}

template<typename T> 
	bool MemoryManager::allocateMemory(std::string &message, Memory_Unit<T> *first)
	{
		return first->allocateMemory(message);
	}

	
///////////////////////////////////////////////////////////
// Memory Deallocate Functions                           //
///////////////////////////////////////////////////////////

template<typename T, typename ...Types>
	bool MemoryManager::deallocateMemory(std::string &message, Memory_Unit<T> *first, Types* ... rest)
	{
		if ( deallocateMemory(message, first) )
			return deallocateMemory(message, rest...);
		else
			return false;
	}

template<typename T>
	bool MemoryManager::deallocateMemory(std::string &message, Memory_Unit<T> *first)
	{
		return first->deallocateMemory(message);
	};
	

///////////////////////////////////////////////////////////
// Copy Device to Host Functions                         //
///////////////////////////////////////////////////////////
	
template<typename T, typename ...Types>
	bool MemoryManager::copyDeviceToHost(std::string &message, Memory_Unit<T> *first, Types* ... rest)
	{
		if (copyDeviceToHost(message, first))
			return copyDeviceToHost(message, rest...);
		else
			return false;
	}

template<typename T>
	bool MemoryManager::copyDeviceToHost(std::string &message, Memory_Unit<T> *first)
	{
		return first->copyDeviceToHost(message);
	}

///////////////////////////////////////////////////////////
// Copy Host to Device Functions                         //
///////////////////////////////////////////////////////////
	
template<typename T, typename ...Types>
	bool MemoryManager::copyHostToDevice(std::string &message, Memory_Unit<T> *first, Types* ... rest)
	{
		if ( copyHostToDevice(message, first) )
			return copyHostToDevice(message, rest...);
		else
			return false;
	}

template<typename T>
	bool MemoryManager::copyHostToDevice(std::string &message, Memory_Unit<T> *first)
	{
		return first->copyHostToDevice(message);
	}

