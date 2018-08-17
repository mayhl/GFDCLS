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
//    list of conditions and the following disclaimer.                              //
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
//         into header file and source code fixed issue.                //
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
		inline bool isCopyableType(Type type) { return (type == pinned || type == non_pinned); };
	}
	
};

#endif