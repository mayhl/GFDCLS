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
		if (n_x != other->getSizeX())
			if (n_y != other->getSizeY())
				if (n_z != other->getSizeZ())
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