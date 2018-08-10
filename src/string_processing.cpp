// -------------------------------------------------------------------- //
// Copyright 2016-2018 Michael-Angelo Yick-Hang Lam                     //
//                                                                      //
// The development of this software was supported by the National       //
// Science Foundation (NSF) Grant Number DMS-1211713.                   //
//                                                                      //
// This file is part of GADIT.                                          //
//                                                                      //
// GADIT is free software: you can redistribute it and/or modify it     //
// under the terms of the GNU General Public License version 3 as       //
// published by the Free Software Foundation.                           //
//                                                                      //
// GADIT is distributed in the hope that it will be useful, but WITHOUT //
// ANY WARRANTY; without even the implied warranty of MERCHANTABILITY   //
// or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public     //
// License for more details.                                            //
//                                                                      //
// You should have received a copy of the GNU General Public License    //
// along with GADIT.  If not, see <http://www.gnu.org/licenses/>.       //
// ---------------------------------------------------------------------//

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

	void cudaErrorToString(cudaError error, std::string &message)
	{
		message =  "Type   : " + std::string(cudaGetErrorName(error)) + "\n";
		message += "Message: " + std::string(cudaGetErrorString(error)) + "\n";
	}
};