import Menu from "@/components/shared/header/menu"
import { BellIcon, MailIcon, PlaneIcon, SettingsIcon } from "lucide-react"
import Link from "next/link"

const LandingPage=()=>{
  return (
    <div
    className="bg-cover h-screen px-10 py-5"
     style={{
      background:"url('/images/eds-bg.jpg')"
    }}
    >
<div className="flex justify-between ">
  <div className="flex gap-2">
  <SettingsIcon className="w-6 h-6 text-pink-300 cursor-pointer"/>
  <MailIcon className="w-6 h-6 text-pink-300 cursor-pointer"/>
  <BellIcon className="w-6 h-6 text-pink-300 cursor-pointer"/>
</div>
<div className="flex flex-col items-center">
  <div >
    <Menu/>
  </div>
  <div className="text-white">John Doe</div>
</div>
</div>

<div className="text-right mt-20">
 <h1 className="text-white text-7xl">WebWizard</h1>
 <p className="text-white">Painting the web with your Vision ...</p>
</div>
<div className="flex gap-20 ">

 
        <div className=" rounded-2xl text-white w-40 h-40 transform rotate-45 bg-white/20 border border-white/30 hover:bg-white/20 transition duration-300 shadow-lg">
          <div className="transform -rotate-45 flex items-center justify-center h-full">
           <Link href="/gen-component">
            <p className="text-center font-semibold">Design a Website</p>
           </Link>
          </div>
        </div>
            <div className=" rounded-2xl text-white w-40 h-40 transform rotate-45 bg-white/20 border border-white/30 hover:bg-white/20 transition duration-300 shadow-lg">
          <div className="transform -rotate-45 flex items-center justify-center h-full">
            <Link href="/gen-website">
            <p className="text-center font-semibold">Design a Website Components</p>
            </Link>
          </div>
        </div>
      


</div>
    </div>
  )
}
export default LandingPage