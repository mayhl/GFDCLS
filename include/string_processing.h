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
// String Processing v1.0                                               //
//                                                                      //
// Purpose: Various commands for processing strings.                    //
//                                                                      //
// -------------------------------------------------------------------- //

#ifndef STRING_PROCESSING_H
#define STRING_PROCESSING_H

#include <string>

#include "cuda_runtime.h"

namespace StringProcessing
{
	void indentLines(std::string &str);

	void centerMultiLineString(std::string &str);
	void centerMultiLineString(std::string &str, const size_t extra_padding);

	size_t findMaxLineLength(const std::string &str);

	void addTitleBanner(std::string &title);
	void addSubtitleBanner(std::string &title);

	void cudaErrorToString(cudaError error, std::string &message);

	std::string valueToString(double value);
};

#endif