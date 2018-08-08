#include "include/memory/base_unit.h"

namespace Memory
{
	Base_Unit::Base_Unit(std::string name, Types::Type type, size_t n_x, size_t n_y, size_t n_z) :
		name(name), type(type), n_x(n_x), n_y(n_y), n_z(n_z) {}

	Base_Unit::Base_Unit(std::string name, Types::Type type, dim3 dimensions ) :
		name(name), type(type), n_x(dimensions.x), n_y(dimensions.y), n_z(dimensions.z) {}

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