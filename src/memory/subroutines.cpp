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