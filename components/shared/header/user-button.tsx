
import { Button } from '@/components/ui/button';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';


import { UserIcon } from 'lucide-react';
import Link from 'next/link';

const UserButton = async () => {
  const session =true

  if (!session) {
    return (
      <Button asChild>
        <Link href='/sign-in'>
          {' '}
          <UserIcon /> Sigin
        </Link>
      </Button>
    );
  }
  return (
    <div>
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            className='
            ring-offset-0 ring-0 hover:ring-0
            rounded-full flex items-center bg-gray-300'
            variant='outline'
          >
          Name
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent className='flex flex-col  w-44 ' align='end'>
          <DropdownMenuLabel>Name</DropdownMenuLabel>

          <DropdownMenuItem>
            <Link href='/user/profile'>Profile</Link>
          </DropdownMenuItem>
          <DropdownMenuItem>
            <Link href='/user/order'>Order</Link>
          </DropdownMenuItem>
      

          <DropdownMenuItem>
          
              <Button>Logout </Button>
         
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
    </div>
  );
};

export default UserButton;
