import Image from 'next/image';
import Link from 'next/link';

import Menu from './menu';

const Header = () => {
  return (
    <header className='w-full border-b bg-gradient-to-r from-blue-800 to-cyan-600 text-white py-2'>
      <div className='container flex px-10 items-center justify-between'>
        <div>
       
          <Link href='/' className='flex ml-4'>
            <Image
              src='/images/logo.jpeg'
              alt={`EDS logo`}
              height={48}
              width={48}
              priority={true}
              className='rounded-b-lg'
            />
            <span className='hidden lg:block font-bold text-2xl ml-3'>
            UI-Automation
            </span>
          </Link>
        </div>
  
        <Menu />
      </div>
    </header>
  );
};

export default Header;
