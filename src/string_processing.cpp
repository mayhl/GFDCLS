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

#include "include/string_processing.h"

namespace StringProcessing
{
	void indentLines(std::string &str)
	{
		// Changeable indent length to control the width of output text
		const std::string indent = "  ";
		const std::string newline = "\n";
		const std::string newline_indent = newline + indent;

		// Indent first line
		str.insert(0, indent);
		
		size_t start_pos = 0;
		// Indent after new line characters
		while ((start_pos = str.find(newline, start_pos)) != std::string::npos) 
		{
			// Do not indent after last new line character.
			if ( start_pos == str.length() - 1 ) break;

			str.replace(start_pos, newline.length(), newline_indent );
			start_pos += newline_indent.length();
		}
	}

	void addTitleBanner(std::string &title)
	{
		for (auto & c : title) c = toupper(c);

		const std::string spacing = std::string(8, ' ');
		title = spacing + title + spacing;
		addSubtitleBanner(title);
		title = "\n" + title + "\n";
	}

	void addSubtitleBanner(std::string &title)
	{
		int title_length = title.length();
		std::string banner = std::string(title_length, '-');
		banner += "\n";

		title = banner + title + "\n" + banner;
	}

	void cudaErrorToString(cudaError error, std::string &message)
	{
		message =  "Type   : " + std::string(cudaGetErrorName(error)) + "\n";
		message += "Message: " + std::string(cudaGetErrorString(error)) + "\n";
	}

	std::string valueToString(double value)
	{
		char buffer[50];

		sprintf(buffer, "%.15e", value );
		std::string str = buffer;

		return str;
	}

};