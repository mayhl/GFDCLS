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

#include <iostream>
#include "include/string_processing.h"

namespace StringProcessing
{
	void centerMultiLineString(std::string &str)
	{
		centerMultiLineString(str, 0);
	}

	void centerMultiLineString(std::string &str, const size_t extra_padding)
	{
		const std::string newline = "\n";

		// Testing if string is one line
		size_t start_pos = str.find(newline);
		if (start_pos == std::string::npos)
		{
			std::string padding = std::string(extra_padding, ' ');
			str = padding + str + padding;
			return;
		}
		// If string is not one line, find max line length
		start_pos = 0;
		size_t old_pos = 0;
		size_t max_length = findMaxLineLength(str) + 2*extra_padding;

		int counter = 0;
		while ((start_pos = str.find(newline, start_pos)) )//&& start_pos != std::string::npos)
		{
			size_t length;

			// Special case for last line
			if (start_pos == std::string::npos)
				length = str.length() - old_pos;
			else
				length = start_pos - old_pos;

			// Computing padding lengths
			size_t total_pad = max_length - length;
			size_t left_pad_size;
			size_t right_pad_size;

			if (total_pad % 2 == 0)
			{
				right_pad_size = total_pad / 2;
				left_pad_size = right_pad_size;
			}
			else
			{
				right_pad_size = total_pad / 2;
				left_pad_size = right_pad_size+1;
			}

			std::string left_padding = std::string(left_pad_size, ' ');
			std::string right_padding = std::string(right_pad_size, ' ');

			// Inserting Padding
			str.insert(old_pos + length, right_padding);
			str.insert(old_pos, left_padding);

			if (start_pos == std::string::npos) break;

			start_pos += total_pad+1;
			old_pos = start_pos;

		}

	}

	size_t findMaxLineLength(const std::string &str)
	{
		const std::string newline = "\n";

		// Testing if string is one line
		size_t start_pos = str.find(newline);
		if (start_pos == std::string::npos) return str.length();

		// If string is not one line, find max line length
		start_pos = 0;
		size_t old_pos = 0;
		size_t max_length = 0;

		while ((start_pos = str.find(newline, start_pos)) != std::string::npos)
		{
			size_t length;

			// Special case for last line
			if (start_pos == std::string::npos)
				length = str.length() - old_pos;
			else
				length = start_pos - old_pos + 1;

			if (length > max_length) max_length = length;

			start_pos += newline.length();
			old_pos = start_pos;
		}

		return max_length;

	}

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
		// Making title upper case
		// Note: Added static_cast to remove compiler warning
		for (char & c : title) c = static_cast<char>( toupper(c) );

		const size_t spacing_size = 8;
		centerMultiLineString(title, spacing_size);

		addSubtitleBanner(title);
		title = "\n" + title + "\n";
	}

	void addSubtitleBanner(std::string &title)
	{

		size_t title_length = findMaxLineLength(title);
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
		const int buffer_size = 50;
		char buffer[buffer_size];

		snprintf(buffer, buffer_size, "%.15e", value );
		std::string str = buffer;

		return str;
	}

};