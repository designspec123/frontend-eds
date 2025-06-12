import { create } from "zustand";

const outputStore = (set, get) => ({
    output: null,

    addOutput: (payload) => {
        set((state) => ({
            output: payload,

        }));
    },


    reset: () => {
        set({
            output: null
        });
    },
});

export const useOutputStore = create(outputStore);

