#ifndef MEMORY_BASE_UNIT
#define MEMORY_BASE_UNIT

#include <stdlib.h> 
#include "cuda_runtime.h"
#include <string>
#include <typeinfo>

#include "include/memory/type.h"

namespace Memory
{
	class Base_Unit
	{
		public:
			Base_Unit(std::string name, Types::Type type, dim3 dimensions);
			Base_Unit(std::string name, Types::Type type, size_t n_x , size_t n_y , size_t n_z );

			virtual bool allocateMemory  (std::string &message) = 0;
			virtual bool deallocateMemory(std::string &message) = 0;
			virtual bool copyDeviceToHost(std::string &message) = 0;
			virtual bool copyHostToDevice(std::string &message) = 0;

		private:

		//	virtual void initialize() = 0;
		protected:

			std::string getName();
			Types::Type getType();
			size_t getMemorySize();

			size_t getSizeX();
			size_t getSizeY();
			size_t getSizeZ();

			std::string name;
			Types::Type type;
			size_t memory_size;

			size_t n_x;
			size_t n_y;
			size_t n_z;

			unsigned short int dimensions;
	};
};

#endif