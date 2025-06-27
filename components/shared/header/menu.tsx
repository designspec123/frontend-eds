"use client"
import { Button } from "@/components/ui/button";
import ModeToggle from "./mode-toggle";
import Link from "next/link";

import { BedSingle, FolderCode } from "lucide-react";
import { User, ChevronDown, Settings, LogOut } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
} from "@/components/ui/dropdown-menu";
import { DropdownMenuTrigger } from "@radix-ui/react-dropdown-menu";
import { useEffect, useState } from "react";
import { redirect } from "next/navigation";

export function UserProfile() {
  const [email,setEmail]=useState("")
  useEffect(()=>{
    const user=localStorage.getItem("user")?JSON.parse(localStorage.getItem("user")||""):""
   if(user){
     setEmail(user?.email)
   }
  },[])
  return (
    <DropdownMenu>
      <DropdownMenuTrigger>
        <div className="w-10 h-10 rounded-full bg-gray-300 flex items-center justify-center text-black">
        {email?email?.charAt(0)?.toUpperCase():<User/>}
        </div>
      </DropdownMenuTrigger>
  

       <DropdownMenuContent className='flex flex-col  w-44 ' align='end'>
          <DropdownMenuLabel>{email}</DropdownMenuLabel>
          <DropdownMenuItem>
           
          <a
            href="#"
            className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
          >
            <Settings className="w-4 h-4 mr-2" />
            Settings
          </a>
      
    
          </DropdownMenuItem>
          <DropdownMenuItem>
              <a
          onClick={()=>{
              localStorage.clear()
             window.location.href="/login"
            }}
            className="flex items-center px-4 py-2 text-sm text-gray-700 hover:bg-gray-100"
          >
            <LogOut className="w-4 h-4 mr-2"/>
            Sign out
          </a>
          </DropdownMenuItem>
      

          <DropdownMenuItem>
          
           
         
          </DropdownMenuItem>
        </DropdownMenuContent>
    </DropdownMenu>
  );
}

const Menu = () => {
  return (
    <div className="flex justify-end gap-3">
      <nav className=" md:flex w-full max-w-xs gap-1">
    
        <UserProfile />
      </nav>
    </div>
  );
};

export default Menu;
